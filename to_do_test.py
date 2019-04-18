# coding=gbk
def test_split():
  test='x_21-x_56'
  print(test.split('-')[0])
  print(test.split('-')[1])


def text_ext():
    list1=[1,2]
    list2=[3,4]
    list3=[5,6]
    a=list1+list3
    print(list1)
    print(a)
    print((list1.extend(list3)).extend(list2))

def test2():
    if ((float(2.1) > float(1.2)) & (float(3.4) > float(1.4))):
        print('这个特征留下来吧')
    else:
        print('ew')

if __name__=='__main__':
    #test_split()
    test2()

