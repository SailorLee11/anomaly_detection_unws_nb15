"""
@Time    : 2021/5/18 21:54
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
-------------------------------------------------
@FileName: company.py
@Software: PyCharm
"""
class Company(object):
    """
    魔法函数是__xxx__
    优先使用迭代器、然后使用getitem这个方法;
    迭代器需要实现  __iter__(self)，这个方法
    """
    def __init__(self,employee_list):
        self.employee = employee_list
    def __getitem__(self, item):
        return self.employee[item]
#     迭代器

company = Company(["tom","bob","jane"])
employee = company.employee
print(type(employee))
for em in employee:
    print(em)
# 有了getitem，对象就可以迭代了
for em in company:
    print(em)
print(type(company))
# 如果没有getitem

class MyVector(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __add__(self, other):
        re_vector = MyVector(self.x+other.x,self.y+other.y)
        return re_vector
    def __str__(self):
        return "x:{x},y:{y}".format(x = self.x,y=self.y)

first_vector = MyVector(1,2)
second_vector = MyVector(2,3)

print(first_vector+second_vector)