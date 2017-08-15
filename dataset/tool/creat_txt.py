import os

file_root = "/home/gxd/project/Seg/dataset/seg/generate_defect1/Train_Data_Done"  
#"/home/gxd/Documents/zhangrui/data/3/test"


txt_path = os.path.join(os.path.dirname(file_root),"data.txt")

with open(txt_path,'w') as f:
	for root,dirs,files in os.walk(file_root):
		if len(dirs) != 0 and len(files) != 0:
			print "must all files or all suddirs,not both"
			break
		for subdir in dirs:
			file_list = os.listdir(os.path.join(root,subdir))
			for file in file_list:
				f.write(os.path.join(subdir,file)+'\n')
		for file in files:
			f.write(os.path.join(file)+'\n')
