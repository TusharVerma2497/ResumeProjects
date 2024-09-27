
Important Instructions: 
	We are using relative referencing to include modules among different parts of the Assignment hence to execute the code properly,
	you need to be in the parent directory of the 'src' directory and run the following commands to execute each part
	You should open the terminal in the here in this folder and run comands from here only




for part1 
		python -m src.Part1.part1 src/Part1/input_img.jpg
		
		Second argument is the path to input image
		generated images will be saved in 'part1_generated_images' folder in the executing directory







for part2
		python -m src.Part2.part2 src/Part2/input_img.jpg
		python -m src.Part2.part2 src/Part2/input_img.jpg 12
		
		Second argument is the path to input image
		Third argument is ths color count for quantization (default=8)
		generated images will be saved in 'part2_generated_images' folder in the executing directory
		
		Note:
			Part2 will automatically run part1 first to find the artistic enhancment so input image should be normal image you provide in part1
			
		
		



for part3
		python -m src.Part3.part3 src/Part3/input_img.jpg src/Part3/target_img.jpg
		
		Second argument is the path to input source image that whill go through artistic enhancement and become a source image for this step
		Third argument is the path to target source image for color transfer
		
		Note:
			Part3 will automatically run part1 first to find the artistic enhancement so input image should be the image you provide in part1
			You can provide target image as either RGB or Gray scale, in case of RGB it will be converted to Gray scale.
			
			
			
