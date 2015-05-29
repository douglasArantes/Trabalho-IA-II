import urllib.request

class Utils(object):
    def downloadFile(self, url):
        return urllib.request.urlopen(url)