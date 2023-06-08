import xml.etree.ElementTree as ET

f = dict()
for item in ET.parse("../data.xml").getroot().findall('item'):
    f[item.find('id').text] = list()
    for scale in item:
        if scale.attrib != {}:
            llist = list()
            for i in scale.attrib.values():
                llist.append(i)
            f[item.find('id').text].append(llist)
print(f)
