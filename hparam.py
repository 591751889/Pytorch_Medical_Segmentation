class hparams:



    train_or_test = 'train'
    log_dir = 'log_thy/'
    aug = None
    latest_checkpoint_file = 'checkpoint_latest.pt'
    total_epochs = 50
    epochs_per_checkpoint = 1
    batch_size = 8
    ckpt = None
    init_lr = 0.001
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '2d' # '2d or '3d'
    in_class = 1
    out_class = 1
    best_dice=0.0
    num_workers = 0
    crop_or_pad_size =256,256,1 # if 2D: 256,256,1
    patch_size = 256,256,1 # if 2D: 128,128,1
    best_dice_model_file = 'best_dice_model.pt'
    # for test
    patch_overlap = 4,4,0 # if 2D: 4,4,0

    model_names = ['unet']
    fold_arch = '*.png'

    save_arch = '.png'

    # source_train_dir = './TN3k/trainval-image/'
    # label_train_dir =  './TN3k/trainval-mask/'
    # source_test_dir =  './TN3k/test-image/'
    # label_test_dir =   './TN3k/test-mask/'

    source_train_dir = './DRIVE/training/images_png'
    label_train_dir =  './DRIVE/training/1st_manual_png/'
    source_test_dir =  './DRIVE/test/images_png'
    label_test_dir =   './DRIVE/test/1st_manual_png/'
    source_val_dir = source_test_dir
    label_val_dir = label_test_dir



    output_dir_test = 'results/'