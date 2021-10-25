import os.path as osp
JOINT = {
    'nyu': 14,
    'icvl': 16,
    'msra': 21,
    'hands17': 21,
}

file_dir = osp.dirname(osp.abspath(__file__))

class Config(object):
    gpu_id = 0
    root_dir = file_dir
    data_dir = osp.join(file_dir, 'data/')
    dataset = 'nyu'
    output_dir = osp.join(file_dir, 'output/')
    load_model = osp.join(file_dir, 'results/hourglass_1.pth')

    jt_num = JOINT[dataset]
    cube = [300, 300, 300]
    augment_para = [10, 0.1, 180]

    net = 'hourglass_1' # 'resnet_18'
    downsample = 2 # [1,2,4]
    img_size = 128
    batch_size = 1
    num_workers = 0
    max_epoch = 80
    loss_type = 'MyL1Loss'
    dense_weight = 1.
    coord_weight = 1000.0
    kernel_size = 0.4   # for offset
    lr = 0.001
    print_freq = 30
    vis_freq = 10


opt = Config()

