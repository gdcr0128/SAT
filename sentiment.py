import paddlehub as hub
import pymysql

senta = hub.Module(name="senta_bilstm")

def getSentiment(text):
    test_text = [text]
    input_dict = {"text": test_text}
    results = senta.sentiment_classify(data=input_dict)
    for result in results:
        print(result)
        Val = result['positive_probs'] - result['negative_probs']
        V = format(Val, '.4f')
        return(V)

# cnx = pymysql.connect(user='root', password='445959', host='192.168.190.128', database='STest',use_unicode = 1, charset='utf8')
# cursor= cnx.cursor()
# count = cursor.execute('SELECT title,id from comment')
# results=cursor.fetchall()
# for row in results:
#     print(row)
#     sentiment = getSentiment(row[0])
#     cursor.execute('UPDATE comment set sentiment = "%s" where id = %s' % (sentiment,row[1]))
#     cnx.commit()
# cursor.close()
# cnx.close()

if __name__ == '__main__':
    text = "Asia Pacific stocks were mostly up Wednesday morning, with investors brushing off a higher-than-expected rise in U.S. " \
           "inflation to focus on the global economic recovery from COVID-19. " \
           "China’s Shanghai Composite inched up 0.01% by 11:12 PM ET (3:12 AM GMT) and the Shenzhen Component jumped 1.36%. " \
           "Wednesday’s March trade data, including exports, imports and the trade balance, continued to give Chinese shares a boost." \
           "Further data, including GDP, industrial production and fixed asset investment, is due on Friday." \
           " Credit markets are monitoring a sharp selloff in China Huarong Asset Management Co. Ltd. (HK:2799), " \
           "one of the country’s largest distressed debt managers. " \
           "The selloff triggered concerns that other heavily leveraged borrowers could also stumble"
    test_text = [text]
    input_dict = {"text": test_text}
    results = senta.sentiment_classify(data=input_dict)
    print(results)
