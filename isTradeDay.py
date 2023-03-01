from datetime import datetime, timedelta
import pymysql
import time

cnx = pymysql.connect(user='root', password='445959', host='192.168.190.128', database='STest',use_unicode = 1, charset='utf8')

def main():
	#字符串 “2020-01-01 09:00:00” 转 时间。函数： datetime.strptime
    start_time = datetime.strptime("2012-06-01 09:00:00", "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime("2022-06-01 23:00:00", "%Y-%m-%d %H:%M:%S")
    current_data = start_time.date()
    while current_data <= end_time.date():
    	# 时间 转 字符串 。 函数 strftime
        current_data_str = current_data.strftime("%Y-%m-%d")
        print("current_data = %s " % str(current_data_str))
        cursor= cnx.cursor()
        cursor.execute('SELECT COUNT(*) as num from trade_date where trade_date = "%s"' % str(current_data_str))
        D = cursor.fetchone()[0]
        print(D)
        if D==1:
            cursor.execute(
                'UPDATE comment set tradeDay = "1" where pubdate = "%s"' % str(current_data_str))
        else:
            cursor.execute(
                'UPDATE comment set tradeDay = "0" where pubdate = "%s"' % str(current_data_str))
        cnx.commit()
        current_data = current_data + timedelta(days=1)

def remove0date(datestr):
    b = time.strptime(datestr, "%Y-%m-%d")  # 时间字符串转为时间元组
    c = time.strftime("%Y-%#m-%#d", b)  # 实现日期的格式化
    return c


if __name__ == "__main__":
    main()
