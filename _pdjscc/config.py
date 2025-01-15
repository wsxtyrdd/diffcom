from datetime import datetime


class config:
    CUDA = True
    gpu_list = '2'
    logger = False
    pass_channel = True
    channel = {"type": 'awgn', 'chan_param': 10}

    img_size = (3, 256, 256)
    norm = normalize = False
    # dataset_dir = '/media/D/Dataset/CIFAR10'
    filename = datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    train_data_dir = ['path of train dataset']
    test_data_dir = ["path of test dataset"]

    C = 2
    kdivn = 2/96

    K_P = 1.0
    K_M = 0.01
    K_S = 0.0
    beta = 5.0

    image_dims = (3, 256, 256)
    use_discriminator = False
    gan_loss_type = 'non_saturating'
    discriminator_steps = 1
    generator_steps = 1
    dis_acc = 1.0
    # Parameters Setting for Training
    epochs = 1000
    batch_size = 8
    test_batch_size = 1
    print_step = 50
    test_step = 1000
    g_learning_rate = 1e-4
    d_learning_rate = 1e-4
    distortion_metric = 'MSE'
    multiple_snr = [1]
    save_epoch = 1
    predict = 'The path of the pretrained model'