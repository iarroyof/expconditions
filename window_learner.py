import sh
import re
from string import punctuation


from pdb import set_trace as st

punctuation = re.sub('[/|<|>]+', '', punctuation)

def build_search_tags(tag_file, return_regexs=True):
    tag_list = []
    with open(tag_dict_f) as f:
        for line in f:
            if "name=" in line:  
                tag_list.append(re.search(name_regex, line).group(1))
            else:
                continue

    if not return_regexs:
        return {tag: ("<{}>".format(tag), "<{}/>".format(tag), id) 
                                    for id, tag in enumerate(tag_list)}
    else:
        return {tag: ("<{}>(.*)<{}/>".format(tag, tag), id) 
                                    for id, tag in enumerate(tag_list)}


def build_window_dataset(lines, tagsregexs, winsize=5):

    samples = []
    for line in lines:
        clean = re.sub('['+punctuation+']', '', line)
        clean = re.sub('><', '> <', clean)
        old_list = [i for i in re.findall(r">([^<>]*)</", clean) if i != ' ']
        new_list = [re.sub(' ', '_', item) for item in old_list]
        words = clean
        for old, new in zip(old_list, new_list):
            words = re.sub(old, new, words)

        words = words.split()[1:]
        for idx, word in enumerate(words):
            if word.startswith("<") and ("</" in word):
                tag = re.search("</(.*)>", word).group(1)
                start = idx - min([idx, winsize])
                end = idx + min([idx + len(words) - 1, winsize]) + 1
                sample = ' '.join(words[start:idx] + words[idx + 1:end])
                sample = re.sub('</?(.*)>', '', sample)
                #sample = re.sub('<(.*)>', '', sample)
                if sample != '':
                     samples.append({"text": sample, "label": tag})

    return samples

inxml = "/home/iarroyof/Dropbox/ES_Carlos_Ignacio/xhGCs/paquete-Nacho/ejemplos-etiquetado-xml/GSE54899_family.xml"
tag_dict_f = "/home/iarroyof/Dropbox/ES_Carlos_Ignacio/xhGCs/paquete-Nacho/ejemplos-etiquetado-xml/esquema-gcs.xsd"
# r'<title[^>]*>([^<]+)</title>'
name_regex = """<xs:element name=\"(.*)\">"""
sample_str = "!Sample_growth_protocol_ch1"
sample_regex = "/^" + sample_str + "/p"

# Get the tag dictionary from the XSD schema
tag_regexps = build_search_tags(tag_dict_f)

lines = sh.sed("-n", sample_regex, inxml)

data = build_window_dataset(lines=lines, tagsregexs=tag_regexps, winsize=5)

samples = []
labels = []
for d in data:
    samples.append(d['text'])
    labels.append(d['label'])

print(samples)
print(labels)
