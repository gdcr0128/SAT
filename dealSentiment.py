from datetime import datetime, timedelta
import pymysql
import time
from numpy import *

def remove0date(datestr):
    b = time.strptime(datestr, "%Y-%m-%d")  # 时间字符串转为时间元组
    c = time.strftime("%Y-%#m-%#d", b)  # 实现日期的格式化
    return c

cnx = pymysql.connect(user='root', password='445959', host='192.168.190.128', database='STest',use_unicode = 1, charset='utf8')
cursor= cnx.cursor()

start_time = datetime.strptime("2012-06-01 09:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.strptime("2022-06-01 23:00:00", "%Y-%m-%d %H:%M:%S")
current_data = start_time.date()
while current_data <= end_time.date():
    current_data_str = current_data.strftime("%Y-%m-%d")
    current_data_deal = remove0date(str(current_data_str))
    count = cursor.execute('SELECT sentiment,tradeDay from comment where pubdate = "%s"' %  str(current_data_str))
    if count != 0:
        results = cursor.fetchall()
        sent_list = []
        for row in results:
            sent_list.append(float(row[0]))
        sentiment = mean(sent_list)
        cursor.execute('UPDATE BDIndex set sentiment = "%s", isTradeDay = "%s" where index_date = "%s"' % (
        format(sentiment, '.4f'), row[1], current_data_deal))
        cnx.commit()
    else:
        sentiment = 0
        cursor = cnx.cursor()
        cursor.execute('SELECT COUNT(*) as num from trade_date where trade_date = "%s"' % str(current_data_str))
        D = cursor.fetchone()[0]
        istradeday = 0
        if D == 1:
            istradeday = 1
        cursor.execute('UPDATE BDIndex set sentiment = "%s", isTradeDay = "%s" where index_date = "%s"' % (
        format(sentiment, '.4f'), istradeday, current_data_deal))
        cnx.commit()
    current_data = current_data + timedelta(days=1)
    # cursor.execute('UPDATE BDIndex set sentiment = "%s", isTradeDay = "%s" where index_date = "%s"' % (format(sentiment, '.4f'), row[1] , current_data_deal))
cursor.close()
cnx.close()
