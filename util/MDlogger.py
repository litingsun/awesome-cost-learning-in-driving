import matplotlib.pyplot as plt
import time 
import os
class Docfig:
    def __init__(self, filename, title=""):
        self.filename = filename
        self.title = title
    def write(self):
        return "\n![{}]({})\n".format(self.title,self.filename)

class Docparagraph:
    def __init__(self, content):
        self.content = content
    def write(self):
        return "\n{}\n".format(self.content)


class Doc:
    """
    use the class to record the items that we want to generate
    basic usage:
        doc = Doc()
        plt.plot(......)
        doc.addplt("figure name") # save the figure into img folder
        doc.generate() # generate corresponding Markdown paragraph
    """
    def __init__(self, logDir = "../devel_log/experiment_results", filename="log_"+time.strftime("%m%d%h")+".md"):
        self.filename = filename
        os.makedirs(logDir, exist_ok=True)
        self.logdir = logDir
        self.items = []
    
    def clear(self):
        self.items = []
    
    def addplt(self,name=""):
        if('.' in name):
            savename = name
        else:
            savename = os.path.join("imgs",name+str(int(time.time()*1000)) + ".png")
        print("\n![{}]({})\n".format(name,savename))

        os.makedirs(os.path.dirname(os.path.join(self.logdir,savename)),exist_ok=True)
        plt.savefig(os.path.join(self.logdir, savename))
        self.items.append(Docfig(savename,name))
    
    def addparagraph(self,content):
        print(content)
        self.items.append(Docparagraph(content))

    def generate(self):
        savename = os.path.join(self.logdir,self.filename)
        print(savename)
        with open(savename,"w") as f:
            for i in self.items:
                f.write(i.write())

        