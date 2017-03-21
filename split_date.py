#-*-coding:utf-8-*-
'''
将usr_log按用户、商户和类别拆分为多个文件
'''

import csv
import os

'''
#记录已存在的date.csv
date_dictionary = {}
   
#将words写入date.csv文件最后一行，文件打开采用'a'模式，即在原文件后添加（add）
def writeByDate(date,words):
    file_name = date+".csv"
    os.chdir('../data/date/')
    if not date_dictionary.has_key(date):
        date_dictionary[date] = True
        f = open(file_name,'a')
        write = csv.writer(f)
        write.writerow(['user_id','item_id','behavior_type','user_geohash','item_category','hour'])
        write.writerow(words)
        f.close()
    else:
        f = open(file_name,'a')
        write = csv.writer(f)
        write.writerow(words)
        f.close()
    os.chdir('../../preprocess/')


#主函数
def splitByDate():
#    os.mkdir('../data/date')
    f = open("../data/train_format1.csv")
    rows = csv.reader(f)
    rows.next()
    for row in rows:
        date = row[-1].split(" ")[0]
        hour = row[-1].split(" ")[1]
        words = row[0:-1]
        words.append(hour)
        writeByDate(date,words)
'''


def writeFile(files, outpath, types):
    x_type = ''
    for rowdata in files:
        if types == "usr":
            x_type = rowdata[0]
        if types == "cate":
            x_type = rowdata[2]
        if types == "seller":
            x_type = rowdata[3]
        
        if x_type != '':
            outfile_write = csv.writer(open(outpath.format(x_type), 'ab'))
            outfile_write.writerow(rowdata)
        
        
        
    #openfile.close()


     
if __name__ == "__main__":
    files_input = "../data/user_log_format1.csv"
    files_output_usr = "../data/usr_id/{}.csv"
    files_output_seller_id = "../data/seller_id/{}.csv"
    files_output_cate = "../data/category/{}.csv"
    
    log_files = csv.reader(open(files_input, 'r'))
    log_files.next()
    
    #按用户id进行分类
    log_files = csv.reader(open(files_input, 'r'))
    log_files.next()
    writeFile(log_files, files_output_usr, "usr")
    
    log_files = csv.reader(open(files_input, 'r'))
    log_files.next()
    writeFile(log_files, files_output_seller_id, "seller")
    
    log_files = csv.reader(open(files_input, 'r'))
    log_files.next()
    writeFile(log_files, files_output_cate, "cate")
    

