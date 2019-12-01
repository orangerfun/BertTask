rawfilepath = "./train.txt"
tarfilepath = "train.tsv"
with open(rawfilepath, "r", encoding="utf8") as fr:
	with open(tarfilepath, "w", encoding="utf8") as fw:
		lines = fr.readlines()
		i = 0
		for line in lines:
			temp = line.split("\t")
			if temp[0] == "体育":
				temp[0] = "0"
			elif temp[0] == "娱乐":
				temp[0] = "1"
			elif temp[0] == "家居":
				temp[0] = "2"
			elif temp[0] == "房产":
				temp[0] = "3"
			elif temp[0] == "教育":
				temp[0] = "4"
			elif temp[0] == "时尚":
				temp[0] = "5"
			elif temp[0] == "时政":
				temp[0] = "6"
			elif temp[0] == "游戏":
				temp[0] = "7"
			elif temp[0] == "科技":
				temp[0] = "8"
			elif temp[0] == "财经":
				temp[0] = "9"
			fw.write("\t".join(temp))
			i += 1
			if i%2000==0:
				print("have write %d lines"%i)


