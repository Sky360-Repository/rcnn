import cv2  # opencv
import os
import xml.etree.ElementTree as ET  # elementpath
from dense_optical_flow import DenseOpticalFlow
import shutil
from pathlib import Path


START_BOUNDING_BOX_ID = 1


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.
    
    Arguments:
        xml_files {list} -- A list of xml file paths.
    
    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        filename = filename[5:]
        return int(filename)
    except:
        raise ValueError(
            "Filename %s is supposed to be an integer." % (filename))


def convert(xml_files, json_file):
    json_dict = {"images": [], "type": "instances",
                 "annotations": [], "categories": []}
    categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


def get_or_create_dir(path, dir):
    full_path = os.path.join(path, dir)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    return full_path


input_dir = '../data/PESMOD/'
output_dir = 'PESMOD'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

#dirs = ['train', 'test']
dirs = ['test']

for tldir in dirs:
    print(tldir)
    input_top_path = os.path.join(input_dir, tldir)
    output_top_path = get_or_create_dir(output_dir, tldir)
    output_annotations_path = get_or_create_dir(output_top_path, 'annotations')
    output_images_path = get_or_create_dir(output_top_path, 'images')
    output_optical_flow_path = get_or_create_dir(
        output_top_path, 'optical_flow')

    file_number = 0

    for video_name in os.listdir(input_top_path):

        video_path = os.path.join(input_top_path, video_name)

        input_annotations_path = os.path.join(video_path, 'annotations')
        annotations = list(sorted(os.listdir(input_annotations_path)))
        full_path_annotations = map(lambda annotation: os.path.join(
            input_annotations_path, annotation), annotations)
        print(get_categories(full_path_annotations))

        input_images_path = os.path.join(video_path, 'images')
        images = list(sorted(os.listdir(input_images_path)))

        dof = None
        for image_filename in images:
            image_full_name = os.path.join(
                input_images_path, image_filename)
            annotation_full_name = os.path.join(
                input_annotations_path, f"{image_filename.split('.')[0]}.xml")

            image = cv2.imread(image_full_name)
            if not dof:
                shape = image.shape
                dof = DenseOpticalFlow(shape[1], shape[0])
                grey_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                dof.process_grey_frame(grey_frame)
                continue
            grey_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            optical_flow_image = dof.process_grey_frame(grey_frame)

            output_file_basename = f"{file_number:06}"
            output_file_with_extension = f"{output_file_basename}.jpg"

            output_optical_flow_full_name = os.path.join(
                output_optical_flow_path, output_file_with_extension)
            cv2.imwrite(output_optical_flow_full_name, optical_flow_image)

            output_image_full_name = os.path.join(
                output_images_path, output_file_with_extension
            )
            shutil.copyfile(image_full_name, output_image_full_name)

            output_annotations_full_name = os.path.join(
                output_annotations_path, f"{output_file_basename}.xml"
            )
            txt = Path(annotation_full_name).read_text()
            txt = txt.replace(image_filename, output_file_with_extension)

            output_path = Path(output_annotations_full_name)
            output_path.write_text(txt)

            file_number += 1
