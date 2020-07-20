import os

class Config():
    def __init__(self):
        super(Config).__init__()

        self.batch_size=128
        self.crop_size = 224
        self.data_name = 'charades_precomp'
        self.data_path = '/media/datasets/ld_data/Charades-STA/charades_c3d/CHARADES_C3D/'
        self.no_imgnorm = False
        self.vocab_path = './vocab/'
        self.vocab_size = 11755

        self.cnn_type='vgg19'
        self.margin = 0.1
        self.max_violation = False
        self.measure = 'cosine'
        self.embed_size=1024
        self.finetune=False
        self.grad_clip=2.0
        self.img_dim=4096
        self.word_dim = 300

        self.learning_rate=0.0001
        self.log_step=10
        self.lr_update = 15
        # self.num_epochs = 40
        self.num_layers = 1
        self.use_abs = False
        self.use_restval = False
        self.val_step = 500
        self.train_epoches = 100
        self.train_test = True

        self.no_val = True

        self.workers = 10
        self.cuda_devices = '0'
        self.logger_name='runs/test_cross_attn_charades_c3dvse7_slide2'
        self.base_path = '/media/datasets/ld_data/Weak_Supervised_Moment-20191023T033504Z-001/Weak_Supervised_Moment'
        self.resume=self.base_path + '/runs/reimplementation/'
        self.resume_developer_last = 'runs/reimplementation/'

        self.platform_lst = ['lab', 'developer', 'cluster']
        self.platform_option = 1


if __name__ == "__main__":
    config = Config()
    print(config.__dict__)



