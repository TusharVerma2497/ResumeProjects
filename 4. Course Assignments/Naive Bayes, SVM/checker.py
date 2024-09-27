import subprocess
import os

parts_list = [['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c'], ['a', 'b', 'c', 'd']]
# parts_list = [[], ['a'], ['a', 'c', 'd']]

folders = os.listdir()
for i in range(1,4):
	print(f'\nChecking Q{i}')
	if f'Q{i}' not in folders:
		print(f'No Q{i} found')
		continue
	os.chdir(f'./Q{i}')
	
	sub_folders = os.listdir()
	for x in parts_list[i-1]:
		print(f'Checking Q{i}/({x})')
		if f'Q{x}' not in sub_folders:
			print(f'No Q{x} folder found')
			continue
		os.chdir(f'./Q{x}')

		files = os.listdir()
		if f'q{x}.py' in files:
			# train = f'../../data/Q{i}/train'
			train = f'/home/tusharverma/MEGA/MEGAsync/MSR at IITD/First Sem/Courses/Machine Learning COL774/Assignments/Assignment2/Submittion/Q1/data/Q{i}/train'
			# test = f'../../data/Q{i}/test'
			test = f'/home/tusharverma/MEGA/MEGAsync/MSR at IITD/First Sem/Courses/Machine Learning COL774/Assignments/Assignment2/Submittion/Q1/data/Q{i}/test'
			subprocess.run(['python',f'q{x}.py',train, test])
		else:
			print(f'No q{x}.py found in folder Q{i}/Q{x}')
		os.chdir('../')
	os.chdir('../')
