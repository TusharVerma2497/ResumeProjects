from ..Part1.lib.ShadowMap import rgb_to_hsi, r_map, computeThreshold, computeShadowMap, merge
from ..Part1.lib.LineDraft import computelineDraft,toGrayScale, bilateral_filtering_slow, bilateral_filtering_fast, applySobel
from ..Part1.lib.colorAdjustment import getChromaticMap, combineShadowAndCMapImages, artisticRendering
import sys
import numpy as np
from ..Part1.lib.helper import changeRange, showImage, readImage, normalizeBy255, saveImage, LABtoRGB



# optimal hyperparameters
shadowImage_lambda=0.3
bilateralFilter_kernel_size=6
bilateralFilter_intensitySigma=45
bilateralFilter_spacialSigma=50
lineDraft_threshold=11
ShadowAndImage_lambda=0.27
ShadowAndCMap_lambda=0.021
artisticRendering_beta=0.1


def artisticEnhancement(img_path, saveImages=False):
    #Fetching the image from system arguments
    print(img_path)
    image = readImage(img_path)
    image=normalizeBy255(image)


    # # Task 1 Generating Shadow map of the image

    h,_,i = rgb_to_hsi(image)
    h= changeRange(h,0,1)
    i= changeRange(i,0,1)

    #computing rmap
    rmap=r_map(h,i)
    # changing the range to 0-255
    rmap=changeRange(rmap,0,255).astype(np.uint8)

    # compute Threshold
    T=computeThreshold(rmap)
    print(f'Shadow Map Threshold computed: {T}')

    # compute shadow map
    sm=computeShadowMap(rmap, T)

    # displaying the shadowed image with best ShadowAndImage_lambda parameter
    shadowedImage=merge(1-sm, image, ShadowAndImage_lambda)





    # # Task 2 Generating Line Draft

    # converting to gray scale
    grayImage=toGrayScale(image)
    grayImage=changeRange(grayImage,0,255).astype(np.uint8)

    # Image, after bilateral filtering
    filteredImage=bilateral_filtering_fast(grayImage,
                            bilateralFilter_kernel_size,
                            bilateralFilter_intensitySigma,
                            bilateralFilter_spacialSigma)

    # applying soble filters
    Ix,Iy=applySobel(filteredImage)
    Ix=Ix.astype(np.float64)
    Iy=Iy.astype(np.float64)

    #getting the edge map
    edgeMap=np.sqrt(np.power(Ix,2),np.power(Iy,2))
    edgeMap=changeRange(edgeMap,0,255).astype(np.uint8)

    #final line draft with lineDraft_threshold
    line=computelineDraft(edgeMap,lineDraft_threshold)





    # # Task 3 Color Adjustment

    #generating the chromatic map
    CMap=getChromaticMap(img_path)

    # displaying with best ShadowAndCMap_lambda value
    enhancedImage=combineShadowAndCMapImages(shadowedImage, CMap, ShadowAndCMap_lambda)

    # final image with best artisticRendering_beta parameter
    finalImage=artisticRendering(1-line,enhancedImage,artisticRendering_beta)
    finalImage=changeRange(finalImage,0,255).astype(np.uint8)
    

    if(saveImages):
        saveImage(rmap.reshape(rmap.shape[0],rmap.shape[1]), name='R_MAP_')
        saveImage(changeRange(sm,0,255).astype(np.uint8).reshape(sm.shape[0],sm.shape[1]), name='Shadow_MAP_')
        saveImage(changeRange(shadowedImage,0,255).astype(np.uint8), name='Shadowed_img_')
        saveImage(filteredImage, name='Bilateral_filtering_img_')
        saveImage(changeRange(Ix,0,255).astype(np.uint8).reshape(Ix.shape[0],Ix.shape[1]), name='Sobel_X_')
        saveImage(changeRange(Iy,0,255).astype(np.uint8).reshape(Iy.shape[0],Iy.shape[1]), name='Sobel_Y_')
        saveImage(edgeMap, name='edge_MAP_')
        saveImage(changeRange(1-line,0,255).astype(np.uint8).reshape(line.shape[0],line.shape[1]), name='Line_draft_')
        saveImage(LABtoRGB(CMap), name='Chromatic_MAP_')
        saveImage(finalImage, name='OUTPUT_ARTISTIC_PART1_')
        saveImage(changeRange(enhancedImage,0,255).astype(np.uint8), name='Enhanced_img_')
    return finalImage


if __name__ == "__main__":

    #invoking part1
    img=artisticEnhancement(img_path=sys.argv[1],saveImages=True)
    showImage(img, 'Artistic Enhancement')
