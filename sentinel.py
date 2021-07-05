import numpy as np
import matplotlib.pyplot as plt
import rasterio as rs


def create_image_and_mask(truecolor_file_name, classification_file_name):
    image = create_image(truecolor_file_name)
    mask = create_mask(classification_file_name)
    return np.append(image, mask, axis=2) 


def create_image(file_name):
    data = rs.open(file_name)
    arr1 = data.read(1)
    arr2 = data.read(2)
    arr3 = data.read(3)
    arr1 = np.expand_dims(arr1, axis=2)
    arr2 = np.expand_dims(arr2, axis=2)
    arr3 = np.expand_dims(arr3, axis=2)
    tmp = np.append(arr1, arr2, axis=2)
    tmp = np.append(tmp, arr3, axis=2)
    return tmp


def create_mask(file_name):

    input = create_image(file_name)

    classes = {
        0: ['nodata', 'defective_pixel', 'shadows', 'cloud_shadows', 'unclassified', 'cloud_medium_probability', 'cloud_high_probability', 'thin_cirrus', 'vegetation', 'not_vegetated'],
        1: ['water'],
        2: ['snow_or_ice']
    }

    esa_classes = {
        'nodata': (0,0,0),
        'defective_pixel': (255,0,0),
        'shadows': (47,47,47),
        'cloud_shadows': (100,50,0),
        'vegetation': (0,161,0),
        'not_vegetated': (255,231,90),
        'water': (0,0,255),
        'unclassified': (129,129,129),
        'cloud_medium_probability': (193,193,193),
        'cloud_high_probability': (255,255,255),
        'thin_cirrus': (100,201,255),
        'snow_or_ice': (255,151,255)
    }

    color_to_class = np.zeros((256, 256, 256), dtype=np.uint8)
    for id in classes.keys():
        for esa_class in classes[id]:
            r, g, b = esa_classes[esa_class]
            color_to_class[r,g,b] = id

    output = np.zeros((input.shape[0], input.shape[1], 1), dtype=np.uint8)

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            r, g, b = input[i,j,:]
            output[i,j,0] = color_to_class[r,g,b]

    return output


def chunks_to_image(chunks, image_size=(1160, 2500)):
    chunk_size = chunks.shape[1]
    image_shape_0 = image_size[0] // chunk_size * chunk_size
    image_shape_1 = image_size[1] // chunk_size * chunk_size
    image = np.zeros((image_shape_0, image_shape_1, chunks.shape[3]), dtype=np.uint8)
    right_ids_0 = np.arange(chunk_size, image_shape_0+1, chunk_size)
    right_ids_1 = np.arange(chunk_size, image_shape_1+1, chunk_size)
    for i, id0 in enumerate(right_ids_0):
        for j, id1 in enumerate(right_ids_1):
            image[id0-chunk_size:id0, id1-chunk_size:id1, :] = chunks[j+len(right_ids_1)*i,:,:,:]
    return image


def image_to_chunks(image, chunk_size=128):
    right_ids_0 = np.arange(chunk_size, image.shape[0], chunk_size)
    right_ids_1 = np.arange(chunk_size, image.shape[1], chunk_size)
    output = np.zeros((1, chunk_size, chunk_size, image.shape[2]), dtype=np.uint8)
    for id0 in right_ids_0:
        for id1 in right_ids_1:
            img = image[id0-chunk_size:id0, id1-chunk_size:id1, :]
            img = np.expand_dims(img, axis=0)
            output = np.vstack([output, img])
    output = np.delete(output, 0, axis=0)
    return output


def get_numpy_file_name(truecolor_file_name):
    return truecolor_file_name.split('_True_color')[0] + '.npy'


def show(image, cmap='viridis'):
    plt.figure(figsize=(20, 9))
    plt.axis('off')
    plt.imshow(image, cmap=cmap)
    plt.show()
