import os
import os.path as osp
import shutil
train_ids = [1,2,4,5,7,8,11,12,13,14,15,16,18,19,20,22,29,30,35,52,53,55,56,58,59,60,61,62,64,65,70,71,72,73,74,76,77,78,79,91,92,95,98,99,107,109,110,111,113,114,115,118,119,120,121,123,124,126,127,128,131,132,133,135,136,137,140,142,143,147,149,151,154,155,156,157,158,159,160,161,163,164,165,168,169,171,174,175,177,178,179,180,181,182,183,184,186,188,189,193,194,196,197,198,199,200,201,203,205,206,208,209,211,212,213,214,216,217,218,219,220,221,222,224,225,226,227,228,230,231,234,235,240,243,244,245,246,247,248,249,250,251,254,255,256,258,260,261,262,264,265,267,268,270,271,276,277,278,279,280,281,283,284,286,287,288,289,290,292,293,294,295,296,297,298,299,304,305,306,308,309,310,311,313,314,316,317,319,320,321,322,323,324,325,326,327,328,329,330,332,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,456,457,458,459,460,461,462,463,464,465,466,467,468,469,470,471,472,473,474,475,476,477,478,479,480,481,482,483,484,486,487,488,489,490,491,492,493,494,495,496,497,498,499,501,502,503,504,505,506,507,508,509,510,512,513,514,515,516,517,518,519,520,521,522,523,524,525,526,527,528,530,531,532,533,334,335,336,337,338,339,340,341,342,343,344,345,346,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,402,403,404,405,406,407,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433]
test_ids = [6,10,17,21,24,25,27,28,31,34,36,37,40,41,42,43,44,45,49,50,51,54,63,69,75,80,81,82,83,84,85,86,87,88,89,90,93,102,104,105,106,108,112,116,117,122,125,129,130,134,138,139,150,152,162,166,167,170,172,176,185,190,192,202,204,207,210,215,223,229,232,237,252,253,257,259,263,266,269,272,273,274,275,282,285,291,300,301,302,303,307,312,315,318,331,333]
cam_rgb = [1,2,4,5]
cam_i = [3,6]

old_root = '/home/zhiqi/dataset/SYSU'
new_root = '/home/zhiqi/dataset/sysu'


list_dirs = os.walk(old_root)
for root, dirs, files in list_dirs:
	for file in files:
		if file.endswith('.jpg') is False: continue
		fpath = osp.join(root,file)
		pid = osp.basename(osp.dirname(fpath))
		camid = '0'+osp.basename(osp.dirname(osp.dirname(fpath)))[3:]
		newfname = pid+'_'+camid+'_'+file
		if int(camid) in cam_rgb and int(pid) in train_ids:
			save_root = osp.join(new_root,'train')
		elif int(camid) in cam_rgb and int(pid) in test_ids:
			save_root = osp.join(new_root,'test','rgb')
		elif int(camid) in cam_i and int(pid) in train_ids:
			save_root = osp.join(new_root,'train')
		elif int(camid) in cam_i and int(pid) in test_ids:
			save_root = osp.join(new_root,'test','infrared')
		else:continue
		if osp.exists(save_root) is False:
			os.makedirs(save_root)
		shutil.copy(fpath,osp.join(save_root,newfname))