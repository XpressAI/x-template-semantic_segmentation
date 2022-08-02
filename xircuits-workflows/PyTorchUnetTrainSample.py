from argparse import ArgumentParser
from datetime import datetime
from time import sleep
import sys
sys.path.append("C:\\Users\\Fahreza.Alghifari\\Documents\\Github\\template-semantic_segmentation\\xai_components")
from xai_sematic_segmentation.butterfly_util import PrepareButterflyDataset
from xai_sematic_segmentation.training import ReadMaskDataSet
from xai_sematic_segmentation.training import ImageTrainTestSplit
from xai_sematic_segmentation.training import PrepareUnetDataLoader
from xai_sematic_segmentation.training import CreateUnetModel
from xai_sematic_segmentation.training import TrainUnet

def main(args):

    ctx = {}
    ctx['args'] = args

    c_1 = PrepareButterflyDataset()
    c_2 = ReadMaskDataSet()
    c_3 = ImageTrainTestSplit()
    c_4 = PrepareUnetDataLoader()
    c_5 = CreateUnetModel()
    c_6 = TrainUnet()

    c_2.dataset_name.value = 'leedsbutterfly/images'
    c_2.mask_dataset_name.value = 'leedsbutterfly/segmentations'
    c_3.input_str = c_2.dataset
    c_3.split_ratio.value = 0.8
    c_3.image_size.value = (256,256)
    c_4.train_image_folder = c_3.train_image_path
    c_4.test_image_folder = c_3.test_image_path
    c_4.training_image_size.value = (256,256)
    c_5.train_data = c_4.train_loader
    c_5.test_data = c_4.tests_loader
    c_5.learning_rate.value = 0.0001
    c_5.earlystop.value = 15
    c_6.train_data = c_4.train_loader
    c_6.test_data = c_4.tests_loader
    c_6.model = c_5.model
    c_6.optimizer = c_5.optimizer
    c_6.early_stopping = c_5.early_stopping
    c_6.gpu.value = 1
    c_6.no_epochs.value = 5
    c_6.wpath_folder.value = 'xircuits-workflows'
    c_6.model_name.value = 'Unet-model'

    c_1.next = c_2
    c_2.next = c_3
    c_3.next = c_4
    c_4.next = c_5
    c_5.next = c_6
    c_6.next = None

    next_component = c_1
    while next_component:
        is_done, next_component = next_component.do(ctx)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), type=str)
    main(parser.parse_args())
    print("\nFinish Executing")