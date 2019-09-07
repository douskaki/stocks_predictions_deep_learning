from bs4 import BeautifulSoup
from pycookiecheat import chrome_cookies
import pandas as pd
import requests

# Uses Chrome's default cookies filepath by default
url = 'https://www.bloomberg.com/search?query=Facebook&startTime=2015-01-01T00:00:00.000Z&endTime=2017-12-31T00:00:00.00Z&page=1'
#cookies = chrome_cookies(url)
cookies = None


def main():
    get_href_headlines()


def get_href_headlines():

    articles_links = []

    for i in range(0, 200):
        page = i
        url = "https://www.bloomberg.com/search?query=Facebook" + \
              "&startTime=2010-01-01T00:00:00.000Z" + \
              "&endTime=2012-12-31T00:00:00.00Z" + \
              "&page=" + str(page)

        postData = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.13; rv:67.0) Gecko/20100101 Firefox/67.0"
        }

        r = requests.get(url, cookies=cookies, data=postData)
        soup = BeautifulSoup(r.content, "html")
        links = soup.find_all("div", attrs={"class": "search-result"})
        print(str(len(links)) + ' ' + str(page))
        articles_links.append(links)

    return articles_links


def flat_list(articles_links):
    return [item for sublist in articles_links for item in sublist]


def convert_to_dataframe(flat_list):

    df = pd.DataFrame(columns=["Timestamp", "Metadata", "Headline", "URL", "ShortDesc"])

    for link in flat_list:
        timestamp = link.find("time", attrs={"class":"published-at"})['datetime']
        metadata = link.find("a", attrs={"class":"metadata-site"}).text if link.find("a", attrs={"class":"metadata-site"}) else ''
        headline = link.find('h1', attrs={"class":"search-result-story__headline"}).text
        url = link.findAll("a")[-1]['href']
        shortDesc = link.find("div", attrs={"class":"search-result-story__body"}).text
        df = df.append({'Timestamp': timestamp,
                        'Metadata':metadata,
                       'Headline': headline,
                        'URL': url,
                        'ShortDesc': shortDesc
                       }, ignore_index=True)

    return df


def write_to_csv(df, filename):
    df.to_csv(filename)


def concat_dataframes(frames, filename):
    result = pd.concat(frames)
    result = result[['Timestamp', 'Metadata', 'Headline', 'URL', 'ShortDesc']]
    result.head()
    result['Timestamp'] = pd.to_datetime(result['Timestamp'])
    result = result.where((pd.notnull(result)), None)
    result['Metadata'] = result['Metadata'].str.strip()
    result['Headline'] = result['Headline'].str.strip()
    result['URL'] = result['URL'].str.strip()
    result['ShortDesc'] = result['ShortDesc'].str.strip()

    duplicateRowsDF = result[result.duplicated()]
    print("Duplicate Rows except first occurrence based on all columns are : ", len(duplicateRowsDF.index))

    result = result[~result.duplicated()]
    result = result.sort_values(by='Timestamp')
    result = result.reset_index(drop=True)
    result.to_csv(filename, index=False)

    # frames = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    # result = pd.concat(frames)
    # result = result[['Timestamp', 'Metadata', 'Headline', 'URL', 'ShortDesc']]
    # result.head()
    # result['Timestamp'] = pd.to_datetime(result['Timestamp'])
    # result = result.where((pd.notnull(result)), None)
    # result['Metadata'] = result['Metadata'].str.strip()
    # result['Headline'] = result['Headline'].str.strip()
    # result['URL'] = result['URL'].str.strip()
    # result['ShortDesc'] = result['ShortDesc'].str.strip()
    # result.head()
    # duplicateRowsDF = result[result.duplicated()]
    # print("Duplicate Rows except first occurrence based on all columns are : ", len(duplicateRowsDF.index))
    # # print(duplicateRowsDF)
    # result = result[result.duplicated() == False]
    # result.shape
    # result = result.sort_values(by='Timestamp')
    # result = result.reset_index(drop=True)
    # result.head()
    # result.to_csv('/Users/Dimitris/Desktop/all_headlines_Dec2017_to_July2018.csv', index=False)


# f1 = pd.read_csv('/Users/Dimitris/Desktop/tenpages.csv')
# f2 = pd.read_csv('/Users/Dimitris/Desktop/tenTO100pages.csv')
# f3 = pd.read_csv('/Users/Dimitris/Desktop/missing20_75to95_pages.csv')
# f4 = pd.read_csv('/Users/Dimitris/Desktop/headlines_Jan2015_Dec2017_100pages_20190624.csv')
# f5 = pd.read_csv('/Users/Dimitris/Desktop/headlines_Jan2015_Dec2017_100to200pages_20190624.csv')
# f6 = pd.read_csv('/Users/Dimitris/Desktop/headlines_Jan2013_Dec2015_159pages_20190624.csv')
# f7 = pd.read_csv('/Users/Dimitris/Desktop/headlines_Jan2013_Dec2015_160to200pages_20190624.csv')
# f8 = pd.read_csv('/Users/Dimitris/Desktop/headlines_Jan2018_Dec2018_88pages_20190624.csv')
# f9 = pd.read_csv('/Users/Dimitris/Desktop/headlines_Jan2018_Dec2018_88to200pages_20190624.csv')
# f10 = pd.read_csv('/Users/Dimitris/Desktop/headlines_Jan2010_Dec2012_200pages_20190624.csv')


if __name__ == "__main__":
    main()
