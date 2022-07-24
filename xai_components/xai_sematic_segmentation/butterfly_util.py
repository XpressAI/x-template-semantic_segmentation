from xai_components.base import InArg, OutArg, Component, xai_component
import os

@xai_component
class PrepareButterflyDataset(Component):

    def __init__(self):
        self.done = False

        
    def execute(self, ctx) -> None:

        fn = "leedsbutterfly_dataset_v1.0.zip"

        if not os.path.exists(fn):

            print("Downloading Leeds Butterfly dataset...")

            import requests
            url = 'http://www.josiahwang.com/dataset/leedsbutterfly/leedsbutterfly_dataset_v1.0.zip'
            r = requests.get(url, allow_redirects=True)

            open(fn, 'wb').write(r.content)

            print("Leeds dataset successfully downloaded.")

        if not os.path.exists("leedsbutterfly"):

            print("Extracting dataset from zip file...")

            import zipfile
            with zipfile.ZipFile("leedsbutterfly_dataset_v1.0.zip","r") as zip_ref:
                zip_ref.extractall(".")

            print("Leeds dataset successfully extracted.")


        if not os.path.exists("leedsbutterfly/segmentations/butterfly/0010001.png"):

            import shutil

            print("Rearranging dataset for segmentation workflow...")

            paths_to_update = ["leedsbutterfly/segmentations/", "leedsbutterfly/images/"]

            for path in paths_to_update:
                new_path = path + "/butterfly/"
                
                files = os.listdir(path)
                os.mkdir(new_path)

                for file in files:

                    # mask segmentation and original image must have the same filename
                    shutil.move(path + file, new_path + file.replace("_seg0",""))

        print("Leeds butterfly dataset ready!")