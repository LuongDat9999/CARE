import random
import logging
import sys
import argparse
import numpy as np
from utils.metrics import *
from utils.helper import *
from model.care import CARE
from dataloader.dataloader import dataloader
import torch.optim as optim

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))


def evaluate(test_batch, rel2idx, ner2idx, args, test_or_dev):
    steps, test_loss = 0, 0
    total_triple_num = [0, 0, 0]
    total_entity_num = [0, 0, 0]
    if args.eval_metric == "macro":
        total_triple_num *= len(rel2idx)
        total_entity_num *= len(ner2idx)

    if args.eval_metric == "micro":
        metric = micro(rel2idx, ner2idx)
    else:
        metric = macro(rel2idx, ner2idx)

    with torch.no_grad():
        for data in test_batch:
            steps += 1
            text = data[0]
            ner_label = data[1].to(device)
            re_label = data[2].to(device)
            mask = data[3].to(device)
            dist = data[4].to(device)

            ner_pred, re_pred = model(text, mask, dist)
            loss = BCEloss(ner_pred, ner_label, re_pred, re_label)
            test_loss += loss

            entity_num = metric.count_ner_num(ner_pred, ner_label)
            triple_num = metric.count_num(ner_pred, ner_label, re_pred, re_label)

            for i in range(len(entity_num)):
                total_entity_num[i] += entity_num[i]
            for i in range(len(triple_num)):
                total_triple_num[i] += triple_num[i]

        triple_result = f1(total_triple_num)
        entity_result = f1(total_entity_num)

        logger.info("------ {} Results ------".format(test_or_dev))
        logger.info("loss : {:.4f}".format(test_loss / steps))
        logger.info(
            "entity: p={:.4f}, r={:.4f}, f={:.4f}".format(entity_result["p"], entity_result["r"], entity_result["f"]))
        logger.info(
            "triple: p={:.4f}, r={:.4f}, f={:.4f}".format(triple_result["p"], triple_result["r"], triple_result["f"]))

    return triple_result, entity_result, test_loss / steps


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='WEBNLG', type=str,
                        help="NYT, WEBNLG")

    parser.add_argument("--epoch", default=10, type=int,
                        help="number of training epoch")

    parser.add_argument("--hidden_size", default=768, type=int,
                        help="number of hidden neurons in the model")

    parser.add_argument("--share_hidden_size", default=128, type=int,
                        help="number of share hidden neurons in the model")

    parser.add_argument("--dist_emb_size", default=20, type=int,
                        help="number of distance embedding hidden neurons in the model")

    parser.add_argument("--co_attention_layers", default=3, type=int,
                        help="number of co-attention layers in the model")

    parser.add_argument("--batch_size", default=8, type=int,
                        help="number of samples in one training batch")

    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="number of samples in one testing batch")

    parser.add_argument("--do_train", default=True,
                        help="whether or not to train from scratch")

    parser.add_argument("--do_eval", default=True,
                        help="whether or not to evaluate the model")

    # parser.add_argument("--embed_mode", default='bert_cased', type=str,
    #                     help="bert_cased or scibert_cased")

    parser.add_argument("--eval_metric", default="micro", type=str,
                        help="micro f1 or macro f1")

    parser.add_argument("--lr", default=0.0001, type=float,
                        help="initial learning rate")

    parser.add_argument("--weight_decay", default=0, type=float,
                        help="weight decaying rate")

    parser.add_argument("--linear_warmup_rate", default=0.0, type=float,
                        help="warmup at the start of training")

    parser.add_argument("--seed", default=0, type=int,
                        help="random seed initiation")

    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate for input word embedding")

    parser.add_argument("--dropconnect", default=0.1, type=float,
                        help="dropconnect rate for partition filter layer")

    parser.add_argument("--steps", default=50, type=int,
                        help="show result for every 50 steps")

    parser.add_argument("--output_file", default="test", type=str,
                        help="name of result file")

    parser.add_argument("--clip", default=0.25, type=float,
                        help="grad norm clipping to avoid gradient explosion")

    # parser.add_argument("--max_seq_len", default=128, type=int, help="maximum length of sequence")

    parser.add_argument('--coslr', action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    output_dir = "save/" + args.output_file
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger.addHandler(logging.FileHandler(output_dir + "/" + args.output_file + ".log", 'w'))
    logger.info(sys.argv)
    logger.info(args)

    model_file = args.output_file + ".pt"

    with open("data/" + args.data + "/ner2idx.json", "r") as f:
        ner2idx = json.load(f)
    with open("data/" + args.data + "/rel2idx.json", "r") as f:
        rel2idx = json.load(f)

    input_size = 768
    model = CARE(args, input_size, ner2idx, rel2idx)

    model.to(device)

    print_params(model)
    train_batch, test_batch, dev_batch = dataloader(args, ner2idx, rel2idx)

    for batch in train_batch:
        print("ner_label shape:", batch[-2].shape)
        print("re_label shape:", batch[-1].shape)
        print("ner nonzero:", (batch[-2] != 0).sum())
        print("re nonzero:", (batch[-1] != 0).sum())
        break

    if args.do_train:
        logger.info("------Training------")
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.coslr:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epoch // 4) + 1)

        if args.eval_metric == "micro":
            metric = micro(rel2idx, ner2idx)
        else:
            metric = macro(rel2idx, ner2idx)

        BCEloss = loss()
        best_average_dev_f1 = best_average_test_f1 = 0
        best_ner_dev_f1, best_re_dev_f1, best_ner_test_f1, best_re_test_f1 = 0, 0, 0, 0

        for epoch in range(args.epoch):
            steps, train_loss = 0, 0

            model.train()
            for data in train_batch:
                steps += 1
                optimizer.zero_grad()

                text = data[0]
                ner_label = data[1].to(device)
                re_label = data[2].to(device)
                mask = data[3].to(device)
                dist = data[4].to(device)

                ner_pred, re_pred = model(text, mask, dist)
                loss = BCEloss(ner_pred, ner_label, re_pred, re_label)
                loss.backward()

                print("ner_pred[0]:", ner_pred[0])  # nên là list các entity span
                print("re_pred[0]:", re_pred[0])  # nên là list các (h, r, t) triple

                train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=args.clip)
                optimizer.step()
                if args.coslr:

                    scheduler.step(epoch)

                if steps % args.steps == 0:
                    logger.info("Epoch: {}, step: {} / {}, loss = {:.4f}".format
                                (epoch, steps, len(train_batch), train_loss / steps))

            logger.info("------ Training Set Results ------")
            logger.info("loss : {:.4f}".format(train_loss / steps))

            if args.do_eval:
                model.eval()
                logger.info("------ Testing ------")
                dev_triple, dev_entity, dev_loss = evaluate(dev_batch, rel2idx, ner2idx, args, "dev")
                test_triple, test_entity, test_loss = evaluate(test_batch, rel2idx, ner2idx, args, "test")
                average_dev_f1 = dev_triple["f"] + dev_entity["f"]
                average_test_f1 = test_triple["f"] + test_entity["f"]
                if average_test_f1 > best_average_test_f1:
                    best_average_test_f1 = average_test_f1
                    best_ner_test_f1, best_re_test_f1 = test_entity["f"], test_triple["f"]
                if average_dev_f1 > best_average_dev_f1:
                    best_average_dev_f1 = average_dev_f1
                    best_ner_dev_f1, best_re_dev_f1 = dev_entity["f"], dev_triple["f"]
                    torch.save(model.state_dict(), output_dir + "/" + model_file)



