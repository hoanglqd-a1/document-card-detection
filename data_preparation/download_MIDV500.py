import json
import numpy as np
import os
import shutil
import wget
import zipfile
from PIL import Image
from glob import glob

download_links = ['ftp://smartengines.com/midv-500/dataset/01_alb_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/02_aut_drvlic_new.zip',
                  'ftp://smartengines.com/midv-500/dataset/03_aut_id_old.zip',
                  'ftp://smartengines.com/midv-500/dataset/04_aut_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/05_aze_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/06_bra_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/07_chl_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/08_chn_homereturn.zip',
                  'ftp://smartengines.com/midv-500/dataset/09_chn_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/10_cze_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/11_cze_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/12_deu_drvlic_new.zip',
                  'ftp://smartengines.com/midv-500/dataset/13_deu_drvlic_old.zip',
                  'ftp://smartengines.com/midv-500/dataset/14_deu_id_new.zip',
                  'ftp://smartengines.com/midv-500/dataset/15_deu_id_old.zip',
                  'ftp://smartengines.com/midv-500/dataset/16_deu_passport_new.zip',
                  'ftp://smartengines.com/midv-500/dataset/17_deu_passport_old.zip',
                  'ftp://smartengines.com/midv-500/dataset/18_dza_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/19_esp_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/20_esp_id_new.zip',
                  'ftp://smartengines.com/midv-500/dataset/21_esp_id_old.zip',
                  'ftp://smartengines.com/midv-500/dataset/22_est_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/23_fin_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/24_fin_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/25_grc_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/26_hrv_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/27_hrv_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/28_hun_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/29_irn_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/30_ita_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/31_jpn_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/32_lva_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/33_mac_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/34_mda_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/35_nor_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/36_pol_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/37_prt_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/38_rou_drvlic.zip',
                  'ftp://smartengines.com/midv-500/dataset/39_rus_internalpassport.zip',
                  'ftp://smartengines.com/midv-500/dataset/40_srb_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/41_srb_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/42_svk_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/43_tur_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/44_ukr_id.zip',
                  'ftp://smartengines.com/midv-500/dataset/45_ukr_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/46_ury_passport.zip',
                  'ftp://smartengines.com/midv-500/dataset/47_usa_bordercrossing.zip',
                  'ftp://smartengines.com/midv-500/dataset/48_usa_passportcard.zip',
                  'ftp://smartengines.com/midv-500/dataset/49_usa_ssn82.zip',
                  'ftp://smartengines.com/midv-500/dataset/50_xpo_id.zip']

DATADIR = 'D:/datasets/MIDV500'
IMAGE   = os.path.join(DATADIR, 'images')
ANNOT   = os.path.join(DATADIR, 'annotations')
TEMP    = os.path.join(DATADIR, 'temp')

PATH_OFFSET = 40

SUBDIR = ['CA', 'CS', 'HA', 'HS', 'KA', 'KS', 'TA', 'TS']

def read_image(img, annot):
    image = Image.open(img)
    orig_shape = image.size
    quad = json.load(open(annot, 'r'))
    coords = np.array(quad['quad'], dtype=np.int32)
    image = image.resize((image.size[0] // 2, image.size[1] // 2))
    normalized_coords = coords.astype(np.float32)
    normalized_coords[:, 0] = normalized_coords[:, 0] / orig_shape[0]
    normalized_coords[:, 1] = normalized_coords[:, 1] / orig_shape[1]
    if not is_valid(normalized_coords):
        return None, None
    annotation = "0 %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f" % (normalized_coords[0, 0], normalized_coords[0, 1], normalized_coords[1, 0], normalized_coords[1, 1], normalized_coords[2, 0], normalized_coords[2, 1], normalized_coords[3, 0], normalized_coords[3, 1])
    return image, annotation

def is_valid(annot: np.ndarray) -> bool:
    return ((annot < 1) & (annot > 0)).all()

def download_and_unzip():
    if os.path.exists(IMAGE) and os.path.exists(ANNOT):
        shutil.rmtree(IMAGE, ignore_errors=True)
        shutil.rmtree(ANNOT, ignore_errors=True)
        os.mkdir(IMAGE)
        os.mkdir(ANNOT)
    
    if not os.path.exists(DATADIR):
        os.mkdir(DATADIR)
        os.mkdir(IMAGE)
        os.mkdir(ANNOT)
        os.mkdir(TEMP)

    file_idx = 0

    for link in download_links:
        filename = link[PATH_OFFSET:]
        full_filename = os.path.join(TEMP, filename)
        directory_name = os.path.join(TEMP, filename[:-4])

        print(full_filename)

        print('Collect and prepare datasets...')

        if not os.path.exists(directory_name):
            if not os.path.isfile(full_filename):
                # file not found, execute wget download
                print ('Downloading:', link)
                wget.download(link, TEMP)

            # Unzip archives
            with zipfile.ZipFile(full_filename, 'r') as zip_ref:
                zip_ref.extractall(TEMP)
        
        print('Prepare dataset... ', directory_name)
        img_dir_path = directory_name + '/images/'
        annot_dir_path = directory_name + '/ground_truth/'

        # Remove unessesary files
        if os.path.isfile(img_dir_path + filename + '.tif'):
            os.remove(img_dir_path + filename.replace('.zip', '.tif'))
        if os.path.isfile(annot_dir_path + filename + '.json'):
            os.remove(annot_dir_path + filename.replace('.zip', '.json'))

        for images, annotations in zip(SUBDIR, SUBDIR):
            img_list = sorted(glob(img_dir_path + images + '/*.tif'))
            annot_list = sorted(glob(annot_dir_path + annotations + '/*.json'))
            for i, (img, annot) in enumerate(zip(img_list, annot_list)):
                if i % 7 != 0: continue
                image, annotation = read_image(img, annot)
                if image is None: continue
                image.save(IMAGE + '/image' + str(file_idx) + '.png')
                with open(ANNOT + '/image' + str(file_idx) + '.txt', 'w') as f:
                    f.write(annotation)

                file_idx += 1

        # shutil.rmtree(directory_name, ignore_errors=True)
        print('----------------------------------------------------------------------')
    
    # shutil.rmtree(TEMP)


if __name__ == '__main__':
    download_and_unzip()