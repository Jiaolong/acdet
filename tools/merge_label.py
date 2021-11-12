import argparse
import glob
from pathlib import Path

from pcdet.utils import  common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--label_car', type=str, default='/data/kitti/training/velodyne',
                        help='specify the config for demo')
    parser.add_argument('--label_ped', type=str, default='/data/kitti/training/label_2',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--label_cyclist',type=str,default='/data/kitti/training/calib')
    parser.add_argument('--label_merge', type=str, default='/data/kitti/training/calib')
    args = parser.parse_args()

    return args


def main():
    args = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    car_labels=glob.glob(str(Path(args.label_car).resolve())+'/*.txt')
    car_labels.sort()
    if not Path(args.label_merge).exists():
        Path(args.label_merge).mkdir(parents=True,exist_ok=True)
        logger.info("crate label_merge directory successful!")
    for car_label in car_labels:
        frame_id=Path(car_label).stem
        ped_label=Path(args.label_ped)/(frame_id+".txt")
        cyc_label=Path(args.label_cyclist)/(frame_id+".txt")
        merge_label=Path(args.label_merge)/(frame_id+".txt")
        with open(car_label, 'r') as f:
            lines_car = f.readlines()
        assert ped_label.exists(), "ped label {} not exist".format(str(ped_label))
        with open(ped_label, 'r') as f:
            lines_ped = f.readlines()
        lines_car.extend(lines_ped)
        logger.info("merge {} successful".format(frame_id + " ped"))
        # assert cyc_label.exists(), "cyc label {} not exist".format(str(cyc_label))
        # with open(cyc_label,'r') as f:
        #     lines_cyc=f.readlines()
        # lines_car.extend(lines_cyc)
        # logger.info("merge {} successful".format(frame_id + " cyc"))
        with open(merge_label, 'w') as f:
            f.writelines(lines_car)
        logger.info("merge {} successful".format(frame_id))
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
