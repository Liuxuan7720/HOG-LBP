# 将十进制数转换为二进制字符串
def decimal_to_binary(decimal_num):
    binary_str = bin(decimal_num)[2:]

    # 补齐为8位二进制数
    binary_str = binary_str.zfill(8)

    # 将二进制字符串转换为列表，并反转顺序
    binary_list = [int(bit) for bit in binary_str[::-1]]

    # print(binary_list)

    return binary_list


# 统计跳转次数
def count_jump(num):
    s = 0
    binary_list = decimal_to_binary(num)
    for i in range(len(binary_list) - 1):
        s += binary_list[i] ^ binary_list[i + 1]
    return s


# 制作LBP等价表
count, lbp_table = 0, [0] * 256
for i in range(256):
    if count_jump(i) <= 2:
        count, lbp_table[i] = count + 1, count
    else:
        lbp_table[i] = 0
print(lbp_table)