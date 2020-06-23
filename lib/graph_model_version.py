import time
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from lib.utils.metric import warmup_lr_scheduler
from lib.utils.metric import AverageMeter
from lib.evaluation import eval_reall_at_N, eval_obj_img, accuracy
from lib.data_layers.vrd_data_graph_weight import VrdDataLayer_weight
from lib.data_layers.vg_data_graph_weight import VgnomeDataLayer_weight
from lib.graph_component import adjoint_rel, adjoint_rel_weight
import pickle



def train_net_gat(train_data_layer, rel_net, epoch, ops, device, args, writer=None):

    co_prior_pth = "../data/{}/co_prior.pkl".format(args.ds_name)
    with open(co_prior_pth, 'rb') as fco:
        co_prior = pickle.load(fco)

    optimizer = ops['optimizer']
    criterion = ops['criterion']
    rel_net.train()

    losses_rel = AverageMeter()
    time1 = time.time()
    epoch_num = train_data_layer._num_instance / train_data_layer._batch_size

    rec_obj_loss = []
    rec_rel_loss = []
    print("Epochs: [{}], lr: {}".format(epoch, optimizer.param_groups[0]['lr']))
    for step in trange(int(epoch_num)):

        image_blob, boxes, rel_boxes, objFea, unionFea, spaFea, classes, unknown, ix1, ix2, \
        class_embed, rel_labels, rel_so_prior, rel_classes = train_data_layer.forward()
        image_blob = torch.from_numpy(image_blob).to(device=device, dtype=torch.float)
        boxes = torch.from_numpy(boxes).to(device=device, dtype=torch.float)
        rel_boxes = torch.from_numpy(rel_boxes).to(device=device, dtype=torch.float)
        objFea = torch.from_numpy(objFea).to(device=device, dtype=torch.float)
        unionFea = torch.from_numpy(unionFea).to(device=device, dtype=torch.float)
        spaFea = torch.from_numpy(spaFea).to(device=device, dtype=torch.float)

        edge_rel, edge_weight = adjoint_rel_weight(classes, ix1, ix2, co_prior, rel_classes)
        edge_rel = torch.from_numpy(edge_rel).to(device=device, dtype=torch.long)
        edge_weight = torch.from_numpy(edge_weight).to(device=device, dtype=torch.float)

        mixFea = torch.cat((objFea, unionFea), 0)

        ix1 = torch.from_numpy(ix1).to(device=device, dtype=torch.long)
        ix2 = torch.from_numpy(ix2).to(device=device, dtype=torch.long)
        class_embed = torch.from_numpy(class_embed).to(device=device, dtype=torch.float)
        target = torch.from_numpy(rel_labels).to(device=device, dtype=torch.long)
        rel_so_prior = -0.5 * (rel_so_prior + 1.0 / train_data_layer._num_relations)
        rel_so_prior = torch.from_numpy(rel_so_prior).to(device=device, dtype=torch.float)

        rel_score, _ = rel_net(image_blob, boxes, rel_boxes, spaFea, ix1, ix2, class_embed, \
                               unknown, edge_rel, edge_weight, args)
        optimizer.zero_grad()
        loss_rel = criterion((rel_so_prior + rel_score).view(1, -1), target)
        losses_rel.update(loss_rel.item())
        loss_rel.backward()
        optimizer.step()


        if step % args.print_freq == 0:
            time2 = time.time()
            print("TRAIN:%d, Rel LOSS:%f, Time:%s" % (step, \
                                                      losses_rel.avg,
                                                      time.strftime('%H:%M:%S', time.gmtime(int(time2 - time1)))))
            time1 = time.time()
            losses_rel.reset()
            rec_rel_loss.append(losses_rel.avg)

        if writer is not None:
            writer.add_scalar('Loss/train', loss_rel.item(), int(epoch_num) * epoch + step)
            writer.add_scalar('LR/train', optimizer.param_groups[0]['lr'], int(epoch_num) * epoch + step)

    return rec_obj_loss, rec_rel_loss


def test_pre_gat(rel_net, device, args,  writer, epoch=0, topk=70):
    # topk = 1, 70 
    rel_net.eval()
    time1 = time.time()
    res = {}
    rlp_labels_ours  = []
    tuple_confs_cell = []
    sub_bboxes_cell  = []
    obj_bboxes_cell  = []

    if args.ds_name == 'vrd':
        test_data_layer = VrdDataLayer_weight('vrd', 'test', model_type = args.model_type)
    if args.ds_name == 'vg':
        test_data_layer = VgnomeDataLayer_weight('vg', 'test', model_type=args.model_type)

    co_prior_pth = "../data/{}/co_prior.pkl".format(args.ds_name)
    with open(co_prior_pth, 'rb') as fco:
        co_prior = pickle.load(fco)

    for step in trange(int(test_data_layer._num_instance)):
        test_data = test_data_layer.forward_test()
        if(test_data is None):
            rlp_labels_ours.append(None)
            tuple_confs_cell.append(None)
            sub_bboxes_cell.append(None)
            obj_bboxes_cell.append(None)
            continue
        image_blob, boxes, rel_boxes, objFea, unionFea, spaFea, classes, unknown_emb,ix1, ix2, \
            label_embeded, ori_bboxes, rel_classes= test_data

        image_blob = torch.from_numpy(image_blob).to(device=device, dtype=torch.float)
        boxes = torch.from_numpy(boxes).to(device=device, dtype=torch.float)
        rel_boxes = torch.from_numpy(rel_boxes).to(device=device, dtype=torch.float)
        objFea = torch.from_numpy(objFea).to(device=device, dtype=torch.float)
        unionFea = torch.from_numpy(unionFea).to(device=device, dtype=torch.float)
        spaFea = torch.from_numpy(spaFea).to(device=device, dtype=torch.float)
        edge_rel, edge_weight = adjoint_rel_weight(classes, ix1, ix2, co_prior, rel_classes)
        edge_rel = torch.from_numpy(edge_rel).to(device=device, dtype=torch.long)
        edge_weight = torch.from_numpy(edge_weight).to(device=device, dtype=torch.float)

        ix1 = torch.from_numpy(ix1).to(device=device, dtype=torch.long)
        ix2 = torch.from_numpy(ix2).to(device=device, dtype=torch.long)
        mixFea = torch.cat((objFea, unionFea), 0)
        unknown_emb = torch.from_numpy(unknown_emb).to(device=device, dtype=torch.float)
        class_embed = torch.from_numpy(label_embeded).to(device=device, dtype=torch.float)
        obj_class_gt = torch.from_numpy(classes).to(device=device, dtype=torch.long)

        with torch.no_grad():
            rel_score, obj_score = rel_net(image_blob, boxes, rel_boxes, spaFea, ix1, ix2, class_embed, \
                                           unknown_emb, edge_rel, edge_weight, args)


        rlp_labels_im = np.zeros((100, 3), dtype=np.float)
        tuple_confs_im = []
        sub_bboxes_im = np.zeros((100, 4), dtype=np.float)
        obj_bboxes_im = np.zeros((100, 4), dtype=np.float)
        rel_prob = rel_score.data.cpu().numpy()
        
        if topk == 1:
            rel_ind = np.argmax(rel_prob, axis=-1)[:100]
  
            for ii in range(rel_ind.shape[0]):
                tuple_idx = ii
                rel = rel_ind[ii]
                conf = rel_prob[tuple_idx, rel]
                sub_bboxes_im[ii] = ori_bboxes[ix1[tuple_idx]]
                obj_bboxes_im[ii] = ori_bboxes[ix2[tuple_idx]]
                rlp_labels_im[ii] = [classes[ix1[tuple_idx]], rel, classes[ix2[tuple_idx]]]
                tuple_confs_im.append(conf)
        else:
            rel_res = np.dstack(np.unravel_index(np.argsort(-rel_prob.ravel()), rel_prob.shape))[0][:100]

            for ii in range(rel_res.shape[0]):

                rel = rel_res[ii, 1]
                tuple_idx = rel_res[ii, 0]
                conf = rel_prob[tuple_idx, rel]
                sub_bboxes_im[ii] = ori_bboxes[ix1[tuple_idx]]
                obj_bboxes_im[ii] = ori_bboxes[ix2[tuple_idx]]
                rlp_labels_im[ii] = [classes[ix1[tuple_idx]], rel, classes[ix2[tuple_idx]]]
                tuple_confs_im.append(conf)

        if(args.ds_name =='vrd'):
            rlp_labels_im += 1
        tuple_confs_im = np.array(tuple_confs_im)
        rlp_labels_ours.append(rlp_labels_im)
        tuple_confs_cell.append(tuple_confs_im)
        sub_bboxes_cell.append(sub_bboxes_im)
        obj_bboxes_cell.append(obj_bboxes_im)
    res['rlp_labels_ours'] = rlp_labels_ours
    res['rlp_confs_ours'] = tuple_confs_cell
    res['sub_bboxes_ours'] = sub_bboxes_cell
    res['obj_bboxes_ours'] = obj_bboxes_cell

    rec_50  = eval_reall_at_N(args.ds_name, 50, res, use_zero_shot = False)
    rec_50_zs  = eval_reall_at_N(args.ds_name, 50, res, use_zero_shot = True)
    rec_100 = eval_reall_at_N(args.ds_name, 100, res, use_zero_shot = False)
    rec_100_zs = eval_reall_at_N(args.ds_name, 100, res, use_zero_shot = True)

    print('CLS TEST r50:%f, r50_zs:%f, r100:%f, r100_zs:%f'% (rec_50, rec_50_zs, rec_100, rec_100_zs))
    time2 = time.time()
    print("TEST Time:%s" % (time.strftime('%H:%M:%S', time.gmtime(int(time2 - time1)))))

    if writer is not None:
        writer.add_scalar("Test_Pre/Recall50", rec_50, int(epoch))
        writer.add_scalar("Test_Pre/Recall100", rec_100, int(epoch))
        writer.add_scalar("Test_ZS/Recall50", rec_50_zs, int(epoch))
        writer.add_scalar("Test_ZS/Recall50", rec_100_zs, int(epoch))

    return rec_50, rec_50_zs, rec_100, rec_100_zs

def test_rel_gat(rel_net, device, topk, args):
    rel_net.eval()
    time1 = time.time()
    
    with open('../data/%s/test.pkl' % args.ds_name, 'rb') as fid:
        anno = pickle.load(fid, encoding='iso-8859-1')
    # 需要载入相关先验信息
    with open('../data/%s/co_prior.pkl' % args.ds_name, 'rb') as fco:
        co_prior = pickle.load(fco)
    res = {}
    rlp_labels_ours = []
    tuple_confs_cell = []
    sub_bboxes_cell = []
    obj_bboxes_cell = []

    if args.ds_name == 'vrd':
        test_data_layer = VrdDataLayer_weight('vrd', 'test', model_type=args.model_type, proposals_path=args.proposal)
    if args.ds_name == 'vg':
        test_data_layer = VgnomeDataLayer_weight('vg', 'test', model_type=args.model_type, proposals_path=args.proposal)

    predict = []

    for step in trange(test_data_layer._num_instance):
    # for step in tqdm(range(3)):
        test_data = test_data_layer.forward()
        if (test_data is None):
            rlp_labels_ours.append(None)
            tuple_confs_cell.append(None)
            sub_bboxes_cell.append(None)
            obj_bboxes_cell.append(None)
            predict.append(None)
            continue

        image_blob, boxes, rel_boxes, spaFea, classes, unknown_emb, ix1, ix2, \
        label_embeded, ori_bboxes, pred_confs, rel_so_prior = test_data

        image_blob = torch.from_numpy(image_blob).to(device=device, dtype=torch.float)
        boxes = torch.from_numpy(boxes).to(device=device, dtype=torch.float)
        rel_boxes = torch.from_numpy(rel_boxes).to(device=device, dtype=torch.float)
        spaFea = torch.from_numpy(spaFea).to(device=device, dtype=torch.float)

        edge_rel, edge_weight = adjoint_rel_weight(classes, ix1, ix2, co_prior)
        edge_rel = torch.from_numpy(edge_rel).to(device=device, dtype=torch.long)
        edge_weight = torch.from_numpy(edge_weight).to(device=device, dtype=torch.float)

        ix1 = torch.from_numpy(ix1).to(device=device, dtype=torch.long)
        ix2 = torch.from_numpy(ix2).to(device=device, dtype=torch.long)

        unknown_emb = torch.from_numpy(unknown_emb).to(device=device, dtype=torch.float)
        class_embed = torch.from_numpy(label_embeded).to(device=device, dtype=torch.float)

        with torch.no_grad():
            rel_score, obj_score = rel_net(image_blob, boxes, rel_boxes, spaFea, ix1, ix2, class_embed, \
                                       unknown_emb, edge_rel, edge_weight, args)


        anno_img = anno[step]
        rel_prob = rel_score.data.cpu().numpy()
        # why added this item, normalize
        rel_prob += np.log(0.5 * (rel_so_prior + 1.0 / test_data_layer._num_relations))
        rel_len = topk
        # 1, 10, 70
        rlp_labels_im = np.zeros((rel_prob.shape[0] * rel_len, 3), dtype=np.float)
        tuple_confs_im = []
        sub_bboxes_im = np.zeros((rel_prob.shape[0] * rel_len, 4), dtype=np.float)
        obj_bboxes_im = np.zeros((rel_prob.shape[0] * rel_len, 4), dtype=np.float)
        n_idx = 0
        for tuple_idx in range(rel_prob.shape[0]):
            sub = classes[ix1[tuple_idx]]
            obj = classes[ix2[tuple_idx]]
            # 每对取topk个
            # for rel in range(rel_prob.shape[1]):
            if (args.use_obj_prior):
                if (pred_confs.ndim == 1):

                    conf = np.log(pred_confs[ix1[tuple_idx]]) + np.log(pred_confs[ix2[tuple_idx]]) + rel_prob[tuple_idx,
                                                                                                     :]

                else:
                    conf = np.log(pred_confs[ix1[tuple_idx], 0]) + np.log(pred_confs[ix2[tuple_idx], 0]) + rel_prob[
                                                                                                           tuple_idx, :]

            else:
                conf = rel_prob[tuple_idx, :]

            topk_ind = np.argsort(-conf.ravel(), axis=0)[:rel_len]

            for rel in topk_ind.tolist():
                sub_bboxes_im[n_idx] = ori_bboxes[ix1[tuple_idx]]
                obj_bboxes_im[n_idx] = ori_bboxes[ix2[tuple_idx]]
                rlp_labels_im[n_idx] = [sub, rel, obj]
                tuple_confs_im.append(conf[rel])
                n_idx += 1

        if (args.ds_name == 'vrd'):
            rlp_labels_im += 1

        tuple_confs_im = np.array(tuple_confs_im)
        idx_order = tuple_confs_im.argsort()[::-1][:100]
        rlp_labels_im = rlp_labels_im[idx_order, :]
        tuple_confs_im = tuple_confs_im[idx_order]
        sub_bboxes_im = sub_bboxes_im[idx_order, :]
        obj_bboxes_im = obj_bboxes_im[idx_order, :]
        rlp_labels_ours.append(rlp_labels_im)
        tuple_confs_cell.append(tuple_confs_im)
        sub_bboxes_cell.append(sub_bboxes_im)
        obj_bboxes_cell.append(obj_bboxes_im)
    res['rlp_labels_ours'] = rlp_labels_ours
    res['rlp_confs_ours'] = tuple_confs_cell
    res['sub_bboxes_ours'] = sub_bboxes_cell
    res['obj_bboxes_ours'] = obj_bboxes_cell

    rec_50 = eval_reall_at_N(args.ds_name, 50, res, use_zero_shot=False)
    rec_50_zs = eval_reall_at_N(args.ds_name, 50, res, use_zero_shot=True)
    rec_100 = eval_reall_at_N(args.ds_name, 100, res, use_zero_shot=False)
    rec_100_zs = eval_reall_at_N(args.ds_name, 100, res, use_zero_shot=True)

    print('CLS REL TEST r50:%f, r50_zs:%f, r100:%f, r100_zs:%f' % (rec_50, rec_50_zs, rec_100, rec_100_zs))
    time2 = time.time()
    print("TEST Time:%s" % (time.strftime('%H:%M:%S', time.gmtime(int(time2 - time1)))))


    return rec_50, rec_50_zs, rec_100, rec_100_zs
