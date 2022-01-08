import argparse
import argparse
import parser

parser = argparse.ArgumentParser()
# parser.add_argument('--foo', nargs='*', )
# parser.add_argument('--bar', nargs='*')
# parser.add_argument('--baz', nargs ='*')
# arg = parser.parse_args()
# print(arg)

# parser = argparse.ArgumentParser()
# parser.add_argument('--foo', nargs='*')
# parser.add_argument('--bar', nargs='*')
# parser.add_argument('baz', nargs='*')
# arg = parser.parse_args('baz 1 2 --foo a b --bar x y'.split())
# print(arg)
# print('a b, x y, 1 2'.split(',', 2))

parser.add_argument('--list', nargs ='+',action='append', type=int)
arg = parser.parse_args()
# for i in range(len(arg.list)):
#     print(type(arg.list[i]))
print(arg.list)
print(len(arg.list))

# import numpy as np
# std = np.array([0.229, 0.224, 0.225]) 
# # a = array.array('f', [1.5, 3, 10.6])
# print(std)


# import argparse #là thư viện Python dùng để xử lí các tham số được truyền vào command-line

# parser = argparse.ArgumentParser() #tạo ra 1 thể hiện argparse

#add_argment() xác định các tham số được xử lí
# '''Method add_argument() có các tham số quan trọng sau:
# ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest]

# name or flags: đây là tham số đầu tiên được truyền vào method. Gồm optional và positional argument. Optional argument được qui định bởi ‘-f’ (short optional argument) 
# hoặc ‘- -foo’ (long optional argument) và là tham số không bắt buộc. Positional argument bởi ‘name’ hoặc ‘path’ và là tham số bắt buộc

# type – kiểu dữ liệu mà  command-line argument cần được chuyển đổi thành (int, float,…).

# default – giá trị mặc định nếu tham số bị bỏ qua.
# required=True – (optional)  quy định argument được add vào là tham số bắt buộc, kể cả optional argument;
# help: chứa nội dung được in ra khi dùng lệnh -h hoặc – -help. (optional)
# dest: keyword dùng để tham khảo đến đối số được truyền vào thay cho default keyword lấy từ “name or flags“. (optional)'''
# parser.add_argument('x', action = 'store', default=2.0, nargs='?' ,type=float)
# parser.add_argument('y', action = 'store', default=3.0, nargs='?' ,type=float)

# args = parser.parse_args() 
# '''chuyển các tham số của command-line thành các object và gán nó như các attributes của namespace. Có thể được chuyển thành kiểu Dictionary với hàm vars([object]).
# Ex:
# args = vars(ap.parse_args()) 
# print args["name"] 
# print args["age"]'''



# print(args.x + args.y)
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("-a", action="store_const", const=42)

# args = parser.parse_args(["-a"])
# print (args.a)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--foo', help='foo help')
args = parser.parse_args()
