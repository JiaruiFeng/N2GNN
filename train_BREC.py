import torch
import torch.nn as nn
import torch_geometric

from datasets.BRECDataset_no4v_60cfi import BRECDataset, part_dict, part_name
from models.input_encoder import EmbeddingEncoder
from models.model_construction import make_model
import train_utils
import time
from loguru import logger
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


def get_dataset(args):
    time_start = time.process_time()
    path, pre_transform, follow_batch = train_utils.data_setup(args)
    path = path + "_" + str(args.onthefly)

    def node_feature_transform(data):
        if "x" not in data:
            data.x = torch.ones([data.num_nodes, 1]).long()
        return data


    if args.onthefly:
        dataset = BRECDataset(name=args.dataset_name,
                              root=path,
                              transform=train_utils.PostTransform(args.wo_node_feature,
                                                                  args.wo_edge_feature),
                              test_part=args.test_part)
    else:
        dataset = BRECDataset(name=args.dataset_name,
                              root=path,
                              pre_transform=T.Compose([node_feature_transform, pre_transform]),
                              transform=train_utils.PostTransform(args.wo_node_feature,
                                                                  args.wo_edge_feature),
                              test_part=args.test_part)
        pre_transform = lambda x: x


    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")
    return dataset, pre_transform, follow_batch

def get_model(args):
    time_start = time.process_time()
    init_encoder = EmbeddingEncoder(2, args.hidden_channels)
    model = make_model(args, init_encoder)
    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model

def evaluation(run, dataset, model, args, device, pre_transform, follow_batch):

    def T2_calculation(dataset, log_flag=False):
        with torch.no_grad():
            loader = DataLoader(dataset, batch_size=args.batch_size, follow_batch=follow_batch)
            pred_0_list = []
            pred_1_list = []
            for data in loader:
                pred = model(data.to(device)).detach()
                pred_0_list.extend(pred[0::2])
                pred_1_list.extend(pred[1::2])
            X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
            Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
            if log_flag:
                logger.info(f"X_mean = {torch.mean(X, dim=1)}")
                logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
            D = X - Y
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            S = torch.cov(D)
            inv_S = torch.linalg.pinv(S)
            return torch.mm(torch.mm(D_mean.T, inv_S), D_mean)

    time_start = time.process_time()
    cnt = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = nn.CosineEmbeddingLoss(margin=args.MARGIN)

    for i in args.test_part:
        part = part_name[i]
        part_range = part_dict[part]
        logger.info(f"{part} part starting ---")

        cnt_part = 0
        fail_in_reliability_part = 0
        start = time.process_time()

        for id in tqdm(range(part_range[0], part_range[1])):
            logger.info(f"ID: {id}")
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.l2_wd
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            dataset_traintest = [pre_transform(data) for data in dataset[
                id * args.NUM_RELABEL * 2: (id + 1) * args.NUM_RELABEL * 2
            ]]

            dataset_reliability = [pre_transform(data) for data in dataset[
                                   (id + args.SAMPLE_NUM) * args.NUM_RELABEL * 2:
                                   (id + args.SAMPLE_NUM + 1) * args.NUM_RELABEL * 2]]

            model.reset_parameters()
            model.train()
            for _ in range(args.num_epochs):
                traintest_loader = DataLoader(dataset_traintest,
                                              batch_size=args.batch_size,
                                              follow_batch=follow_batch)
                loss_all = 0
                for data in traintest_loader:
                    optimizer.zero_grad()
                    pred = model(data.to(device))
                    loss = loss_func(
                        pred[0::2],
                        pred[1::2],
                        torch.tensor([-1] * (len(pred) // 2)).to(device),
                    )
                    loss.backward()
                    optimizer.step()
                    loss_all += len(pred) / 2 * loss.item()
                loss_all /= args.NUM_RELABEL
                logger.info(f"Loss: {loss_all}")
                if loss_all < args.LOSS_THRESHOLD:
                    logger.info("Early Stop Here")
                    break
                scheduler.step(loss_all)

            model.eval()
            T_square_traintest = T2_calculation(dataset_traintest, True)
            T_square_reliability = T2_calculation(dataset_reliability, True)

            isomorphic_flag = False
            reliability_flag = False
            if T_square_traintest > args.THRESHOLD and not torch.isclose(
                T_square_traintest, T_square_reliability, atol=args.EPSILON_CMP
            ):
                isomorphic_flag = True
            if T_square_reliability < args.THRESHOLD:
                reliability_flag = True

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id)
                logger.info(f"Correct num in current part: {cnt_part}")
            if not reliability_flag:
                fail_in_reliability += 1
                fail_in_reliability_part += 1
            logger.info(f"isomorphic: {isomorphic_flag} {T_square_traintest}")
            logger.info(f"reliability: {reliability_flag} {T_square_reliability}")

        end = time.process_time()
        time_cost_part = round(end - start, 2)

        logger.info(
            f"{part} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        logger.info(
            f"Fail in reliability: {fail_in_reliability_part} / {part_range[1] - part_range[0]}"
        )


    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"evaluation time cost: {time_cost}")

    Acc = round(cnt / args.SAMPLE_NUM, 2)
    logger.info(f"Correct in {cnt} / {args.SAMPLE_NUM}, Acc = {Acc}")

    logger.info(f"Fail in reliability: {fail_in_reliability} / {args.SAMPLE_NUM}")
    logger.info(correct_list)

    logger.add(f"{args.save_dir}/{args.exp_name}/result_show.txt", format="{message}", encoding="utf-8")
    logger.info(
        "Real_correct\tCorrect\tFail\thops\tlayers"
    )
    logger.info(
        f"{cnt-fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{args.num_hops}\t{args.num_layers}"
    )



#TODO: Fit BREC dataset into pytorch_lightning framework.
def main():
    parser = train_utils.args_setup()
    parser.add_argument('--dataset_name', type=str, default="BREC", help='Name of dataset.')
    parser.add_argument('--NUM_RELABEL', type=int, default=32)
    parser.add_argument('--P_NORM', type=int, default=2)
    parser.add_argument('--THRESHOLD', type=float, default=72.34)
    parser.add_argument('--MARGIN', type=float, default=0.0)
    parser.add_argument('--LOSS_THRESHOLD', type=float, default=0.2)
    parser.add_argument('--EPSILON_MATRIX', type=float, default=1e-7)
    parser.add_argument('--EPSILON_CMP', type=float, default=1e-6)
    parser.add_argument('--runs', type=int, default=10, help='Number of repeat run.')
    parser.add_argument('--test_part', type=list, default=range(5), help="A list of index to indicate which part to test.")
    parser.add_argument('--onthefly', action="store_true", help="If true, process the data on the fly to save memory.")

    args = parser.parse_args()
    args = train_utils.update_args(args)
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.SAMPLE_NUM = sum([part_dict[part_name[i]][1]-part_dict[part_name[i]][0] for i in args.test_part])




    # get dataset
    dataset, pre_transform, follow_batch = get_dataset(args)


    for run in range(args.runs):
        seed = train_utils.get_seed(args.seed)
        torch_geometric.seed_everything(seed)
        # get model
        model = get_model(args)
        model.to(device)
        evaluation(run + 1, dataset, model, args, device, pre_transform, follow_batch)


if __name__ == "__main__":
    main()












