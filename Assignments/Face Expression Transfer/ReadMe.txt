
Important Instructions: 
	There is also an anaconda virtual environment file just in case something didn't work
	caching facility has been provided once you have selected the anchor points for a set of {I1,I2,I3,#anchor_points} it will be cached and can be used again
	if you wish to reselect the anchor points kindly delete the chached file for the set, name of the file is displayed in the terminal every time you run the execution command 




for part1 
		
		task{n}: execute in the part1 directory itself
		
			Format:  python task{n}.py images/expression/f1normal.jpg images/expression/f2normal.jpg images/expression/f1happy.jpg x
			Arguments:
				-path to Image I1
				-path to Image I2
				-path to Image I3
				-number of anchor points: x
			
			ex:  python task1.py images/expression/f1normal.jpg images/expression/f2normal.jpg images/expression/f1happy.jpg 3
			generated images will be saved in 'task{n}_results' folder in the executing directory







for part2

		execute in the part2 directory itself
		
		ex:  python part2.py images/expression/f1normal.jpg images/expression/f2normal.jpg 30
		Arguments:
			-path to source face Image 
			-path to target Image 
			-number of anchor points:
		
		generated images will be saved in 'part2_results' folder in the executing directory
	
			
			
