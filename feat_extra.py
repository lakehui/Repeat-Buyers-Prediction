# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 22:45:13 2016

@author: 521-hui
"""

import os
import csv
from collections import Counter
import pickle

'''
用户特征：用户购买的天数，重复购买的次数，购买的次数，收藏次数，加购物车次数，点击天数，年龄，性别

商户：总销量，总共买家数量，重复买家数量，被收藏的总数量，被加购物车总次数

商户&用户：被该用户点击的天数，被该用户收藏的次数，被该用户加购物车的次数，被该用户购买的数量，对应用户所购买的商品其总销量（多件商品取最小、最大、平均值）
            对应用户所购买的商品其总共的重复购买数量（多件商品取最小、最大、平均值）
            用户所购买产品在该类别中的市场份额，该类商品共有多少卖家，该类商品有多少的重复买家，该类商品的总销量
'''

'''
分时段统计，30天，90天，180天
统计类别中重复买家的数量，可以反映该类是不是常用品
'''

#文本转数字，若遇到空白则保留
def getOneLine(row):
    
    linedata = []  
    for i in range(len(row)): 
        if row[i] != '':           
            linedata.append(int(row[i]))
        else:
            linedata.append('')
    return linedata

#==================================from category file=============================#
# sales of products buyed by usr 
def _getSalesTotal(seller_product, cate_logs):

    sales_product = 0
    for row in cate_logs:
        if row[6] == 2 and row[3] == seller_product[0] and row[1] == seller_product[1]:
            sales_product = sales_product + 1
    
    return sales_product

#return: the share of the product in this category
#        the number of seller in the category
#        the repeated buyer in the category
#        the total sales in the category
def genCateDict(tr_samp):       #tr_samp假设已经转为int类型

    cate_feat = {}       #特征变量  用词典保存，键值
    
    #creat key table
    samp_dict = {}
    for row in tr_samp:
        key_tr = str(row[0]) + '+' + str(row[1])
        samp_dict[key_tr] = 0
            
    for filename in os.listdir("../data/category"):
        print 'category'+filename
        
        cate_path = open("../data/category/{}".format(filename), 'r')
        cate_read = csv.reader(cate_path)
        
        isused_dict = {}
        cate_logs = []
        for row in cate_read:
            if int(row[5]) < 1113 or row[5] == '':
                cate_logs.append(getOneLine(row))
        
        cate_path.close()
        
        #number of seller in the category
        seller_cate = [x[3] for x in cate_logs]
        seller_dict = Counter(seller_cate)
        total_seller_cate = len(seller_dict)
        
        #number of the repeated buyer in this cate
        cate_buy_logs = [x for x in cate_logs if x[6] == 2]
        usr_dict = {}
        for row in cate_buy_logs:
            if usr_dict.has_key(row[0]):    #statistic usr and his buy date
                usr_dict[row[0]].append(row[5])      
            else:
                usr_dict[row[0]] = [row[5]]
            
        num_repeat_buyer  = 0
        for value in usr_dict.itervalues():
            if len(Counter(value)) > 1:
                num_repeat_buyer = num_repeat_buyer + 1
                
        #total sales in the cate
        total_sale_cate = len(cate_buy_logs)
        
        #according to training file to statistic the share of product buyed by usr in the category
        for row in cate_logs:
            key_cate = str(row[0]) + '+' + str(row[3])
            key_isused = str(row[0]) + '+' + str(row[3]) + '+' \
                            + str(row[2]) + '+' + str(row[1])
            
            if (samp_dict.has_key(key_cate)) and (not isused_dict.has_key(key_isused)) and (row[6] == 2):
                sales_product = _getSalesTotal([row[3], row[1]], cate_logs)
                #share of product in the category
                cate_feat[key_isused] = [float(sales_product)/total_sale_cate]
                            
                cate_feat[key_isused].append(total_seller_cate)
                cate_feat[key_isused].append(num_repeat_buyer)
                cate_feat[key_isused].append(total_sale_cate)
                
                isused_dict[key_isused] = 1
                
    return cate_feat        #key: 'usr_id+seller_id+cate+product'

#======================================from seller file============================#
def _ClickandCollect(usr_seller, seller_logs):
    
    click_date = []
    collect_logs = []
    cart_logs = []
    buy_logs = []
    for row in seller_logs:
        if row[6] == 0 and usr_seller[0] == row[0] and usr_seller[1] == row[3]:
            click_date.append(row[5])
        if row[6] == 1 and usr_seller[0] == row[0] and usr_seller[1] == row[3]:
            collect_logs.append(row)
        if row[6] == 3 and usr_seller[0] == row[0] and usr_seller[1] == row[3]:
            cart_logs.append(row)
        if row[6] == 2 and usr_seller[0] == row[0] and usr_seller[1] == row[3]:
            buy_logs.append(row)
    
    days_click = len(Counter(click_date))    
    num_collect = len(collect_logs)
    num_cart = len(cart_logs)
    num_buy = len(buy_logs)
    
    return days_click, num_collect, num_cart, num_buy
    

def _repeatedStatistic(obj_stat, obj_buy_logs):
    
    buy_dict = {}
    num_repeated = 0
    for row in obj_buy_logs:
        if buy_dict.has_key(row[obj_stat]):
            buy_dict[row[obj_stat]].append(row[5])
        else:
            buy_dict[row[obj_stat]] = [row[5]]
    
    for value in buy_dict.itervalues():
            if len(Counter(value)) > 1:
                num_repeated = num_repeated + 1
    
    return num_repeated
                
    
def _productSaleandRepeatedBuy(product, seller_logs):
    
    buy_product_logs  = [x for x in seller_logs if x[6] == 2 and x[1] == product]
    
    sales_product = len(buy_product_logs)
    
    num_repeated_product = _repeatedStatistic(0, buy_product_logs)
    
    return sales_product, num_repeated_product
    

def _getUsrProduct(usr, seller_logs):
    
    usr_buy_logs = [x[1] for x in seller_logs if x[0] == usr and x[6] == 2]
    usr_buy_dict = Counter(usr_buy_logs)
    
    usr_buy_product = []
    for key in usr_buy_dict.iterkeys():
        usr_buy_product.append(key)
    
    return usr_buy_product
              
            
def genSellerDict(tr_samp):
    
    seller_dict = {}    #store the feature from seller and usr-seller
    
    #read cate feature from dict_cate file
    cate_file = open("../data/feature/feat_cate.pytmp", 'r')
    cate_feat = pickle.load(cate_file)
    cate_file.close()
       
    #creat key table
    samp_dict = {}
    for row in tr_samp:
        key_tr = str(row[0]) + '+' + str(row[1])
        samp_dict[key_tr] = 0
            
    for filename in os.listdir("../data/seller_id"):
        print 'selle_id'+filename
    
        seller_path = open("../data/seller_id/{}".format(filename), 'r')
        seller_read = csv.reader(seller_path)
        
        isused_dict = {}
        seller_logs = []
        for row in seller_read:
            if int(row[5]) < 1113 or row[5] == '':
                seller_logs.append(getOneLine(row))
        
        seller_path.close()
        
        #the total of user who has buyed in this seller
        usr_log_buy = [x for x in seller_logs if x[6] == 2]
        num_usr_buy = len(Counter([x[0] for x in usr_log_buy]))
        
        #the total sales in this seller
        total_sales = len(usr_log_buy)
        
        #the number of repeated buyer in this seller
        num_repeated_buyer = _repeatedStatistic(0, usr_log_buy)     #0: usr
        
        #the number of collection in this seller
        num_collect_seller = len([x for x in seller_logs if x[6] == 3])
        
        #the number of being added into cart
        num_cart_seller = len([x for x in seller_logs if x[6] == 1])

        for row in seller_logs:
            key_seller = str(row[0]) + '+' + str(row[3])
            key_isused = key_seller
                            
            if (samp_dict.has_key(key_seller)) and (not isused_dict.has_key(key_isused)) and (row[6] == 2):
                #the number of days of clicking, the number of collecting, 
                #           the number of adding into cart, the number of purchase
                days_click_usr, num_collect_usr, num_cart_usr, num_buy_usr = _ClickandCollect([row[0], row[3]], seller_logs)
                
                #the sales(min,max,mean) of the product buyed by usr, the number(min,max,mean) of repeated buy
                usr_product = _getUsrProduct(row[0], seller_logs)
                sales_product = []
                num_repeated_product = []
                for usr_product_id in usr_product:                                     
                    sales_tmp, num_repeated_tmp = _productSaleandRepeatedBuy(usr_product_id, seller_logs)
                    sales_product.append(sales_tmp)
                    num_repeated_product.append(num_repeated_tmp)                    
                    
        #========此处需要更改，用户所该买的类别可能不止一种，也应该使用最大最小均值统计==========================#
                    #category feature between the usr and the seller
                    key_cate = key_seller + '+' + str(row[2]) + '+' + str(row[1])
                    cate_feat_product_usr = cate_feat[key_cate]
                
                if len(sales_product) != 0:
                    min_sales_product = min(sales_product)
                    max_sales_product = max(sales_product)
                    mean_sales_product = float(sum((sales_product)))/len(sales_product)
                else:
                    min_sales_product = 0
                    max_sales_product = 0
                    mean_sales_product = 0
                    
                if len(num_repeated_product) != 0:
                    min_repeated_product = min(num_repeated_product)
                    max_repeated_product = max(num_repeated_product)
                    mean_repeated_producr = float(sum(num_repeated_product))/len(num_repeated_product)
                else:
                    min_repeated_product = 0
                    max_repeated_product = 0
                    mean_repeated_producr = 0
                    

                seller_dict[key_isused] = [total_sales, num_usr_buy, num_repeated_buyer,num_collect_seller, num_cart_seller,\
                                            days_click_usr, num_collect_usr, num_cart_usr, num_buy_usr,\
                                            min_sales_product, max_sales_product, mean_sales_product,\
                                            min_repeated_product, max_repeated_product, mean_repeated_producr] + cate_feat_product_usr
                
                isused_dict[key_isused] = 1
    #商户：总销量，总共买家数量，重复买家数量，被收藏的总数量，被加购物车总次数
    #商户&用户：被该用户点击的天数，被该用户收藏的次数，被该用户加购物车的次数，被该用户购买的数量，对应用户所购买的商品其总销量（多件商品取最小、最大、平均值）
                #对应用户所购买的商品其总共的重复购买数量（多件商品取最小、最大、平均值）
                #用户所购买产品在该类别中的市场份额，该类商品共有多少卖家，该类商品有多少的重复买家，该类商品的总销量
                
    return seller_dict

#============================from user files===========================================================#
#the usr buying days, the number of repeated buy, purchase quantity
def _countDaysbyUsr(usr_logs):
    buy_log = [x for x in usr_logs if x[6] == 2]   
    
    date = [x[5] for x in buy_log if x[5] != '']
    count_date = Counter(date)
    usr_buy_days = len(count_date) 
    
    a = {}
    for row in buy_log:
        if a.has_key(row[3]):
            a[row[3]].append(row[5])
        else:
            a[row[3]] = [row[5]]
    
    num_repeatebuy = 0      
    for value in a.itervalues():
        if len(Counter(value)) > 1:
            num_repeatebuy = num_repeatebuy + 1
        
    num_buys = len(buy_log)
            
    return [usr_buy_days, num_repeatebuy, num_buys]

#num of collection、 num of cart、 online days on taobao
def _countCollectandCart(usr_logs):
    collect_log = [x for x in usr_logs if x[6] == 3]
    cart_log = [x for x in usr_logs if x[6] == 1]
    num_collect = len(collect_log)
    num_cart = len(cart_log)
    
    date = [x[5] for x in usr_logs if x[6] == 0]
    count_date = Counter(date)
    usr_click_days = len(count_date)
    
    return [num_collect, num_cart, usr_click_days]

def genUsrDict():
    
    print "usr feature extract..."
    
    usr_dict = {}
    
    usr_info_dict = {}
    usr_info_file = open("../data/user_info_format1.csv", 'r')
    usr_info_read = csv.reader(usr_info_file)
    for row_str in usr_info_read:
        row = getOneLine(row_str)
        if row[1] == '':
            row[1] = 3
        if row[2] == '' or row[2] == 2:
            row[2] = 0.5
        usr_info_dict[str(row[0])] = [row[1], row[2]] 
    usr_info_file.close()
    
    for filename in os.listdir("../data/usr_id"):
        #print 'usr_id/'+filename
    
        usr_path = open("../data/usr_id/{}".format(filename), 'r')
        usr_read = csv.reader(usr_path)
        
        usr_logs = []
        for row in usr_read:
            if int(row[5]) < 1113 or row[5] == '':
                usr_logs.append(getOneLine(row))
        
        usr_path.close()
        
        usr_buy_days, num_repeatebuy, num_buys = _countDaysbyUsr(usr_logs)
        num_collect, num_cart, usr_click_days = _countCollectandCart(usr_logs)
        
        usr_dict[str(row[0])] = [usr_buy_days, num_repeatebuy, num_buys,\
                                        num_collect, num_cart, usr_click_days] + usr_info_dict[str(row[0])]
                                        
    return usr_dict
        
        
#==============================generate the final feature============================#
#return: 
def genFinalFeat(tr_samp):
    
    print "generate the final feature..."
    
    finalfeat = []
    label = []
    
    #read seller feature from dict_seller file
    seller_file = open("../data/feature/feat_seller.pytmp", 'r')
    seller_feat = pickle.load(seller_file)
    seller_file.close()
    
    #read usr feature from dict_usr file
    usr_file = open("../data/feature/feat_usr.pytmp", 'r')
    usr_feat = pickle.load(usr_file)
    usr_file.close()
       
    #combine seller feature with usr feature
    for row in tr_samp:
        key_tr = str(row[0]) + '+' + str(row[1])
        finalfeat.append(usr_feat[str(row[0])]+seller_feat[key_tr])
        #label.append(row[2])

    #用户特征：用户购买的天数，重复购买的次数，购买的次数，收藏次数，加购物车次数，点击天数，年龄，性别
    #商户：总销量，总共买家数量，重复买家数量，被收藏的总数量，被加购物车总次数
    #商户&用户：被该用户点击的天数，被该用户收藏的次数，被该用户加购物车的次数，被该用户购买的数量，对应用户所购买的商品其总销量（多件商品取最小、最大、平均值）
                #对应用户所购买的商品其总共的重复购买数量（多件商品取最小、最大、平均值）
                #用户所购买产品在该类别中的市场份额，该类商品共有多少卖家，该类商品有多少的重复买家，该类商品的总销量
    return finalfeat#, label

if __name__ == "__main__":
    '''
    trainfile = open("../data/test_format1.csv", 'r')
    train_read = csv.reader(trainfile)
    train_read.next()
    
    train_samp = [getOneLine(row) for row in train_read]
    trainfile.close()
    
    cate_feat = genCateDict(train_samp)    
    save_cate = open("../data/feature/feat_cate.pytmp", 'wb')
    pickle.dump(cate_feat, save_cate)
    save_cate.close()
    
    seller_feat = genSellerDict(train_samp)
    save_seller = open("../data/feature/feat_seller.pytmp", 'wb')
    pickle.dump(seller_feat, save_seller)
    save_seller.close()
    
    usr_feat = genUsrDict()
    save_usr = open("../data/feature/feat_usr.pytmp", 'wb')
    pickle.dump(usr_feat, save_usr)
    save_usr.close()
    
    #train
    feature_all, label = genFinalFeat(train_samp)
    save_feat = open("../data/feature/feat_final.pytmp", 'wb')
    save_label = open("../data/feature/label.pytmp", 'wb')
    pickle.dump(feature_all, save_feat)
    pickle.dump(label, save_label)
    save_feat.close()
    save_label.close()
    '''
    #test
    feature_all = genFinalFeat(train_samp)
    save_feat = open("../data/feature/feat_final.pytmp", 'wb')
    pickle.dump(feature_all, save_feat)                
    save_feat.close()