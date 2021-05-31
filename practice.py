# def summ(n,m):
#     if n <= 0:return []
#     def dfs(start_index,target,res,path):
#         if target == 0:
#             res.append(path[:])
#         for i in range(start_index,n):
#             if i <= target:
#                 path.append(i)
#                 dfs(i+1,target-i,res,path)
#                 path.pop()
#     res = []
#     dfs(0,m,res,[])
#     return res
# test = summ(5,3)
# print(test)


# def permute(nums):
#     if not nums: return []
#     visted = [False for _ in range(len(nums))]
#
#     def dfs(visted,nums, res, path, depth):
#         if depth == len(nums):
#             res.append(path[:])
#             return res
#         for i in range(len(nums)):
#             if not visted[i]:
#                 visted[i] = True
#                 path.append(nums[i])
#                 dfs(visted,nums, res, path, depth + 1)
#                 visted[i] = False
#                 path.pop()
#
#     res = []
#     dfs(visted,nums, res, [], 0)
#     return res
def permute2(nums):
    if not nums: return []
    nums.sort()
    visted = [False for _ in range(len(nums))]
    def dfs(visted,nums, res, path, depth):
        if depth == len(nums):
            res.append(path[:])
            return res
        for i in range(len(nums)):
            if not visted[i]:
                if i > 0 and nums[i] == nums[i-1] and not visted[i-1]:continue
                visted[i] = True
                path.append(nums[i])
                dfs(visted,nums, res, path, depth + 1)
                visted[i] = False
                path.pop()

    res = []
    dfs(visted,nums, res, [], 0)
    return res
nums = [1,2,1]
test = permute2(nums)
print(test)

def subkarray(nums,k):
    i,res,tmp = 0,0,0
    for j in range(len(nums)):
        res += nums[j]
        while res > k:
            res -= nums[i]
            i += 1
        tmp += j-i+1
    return tmp
nums = [1,2,3,1,1,2]
test = subkarray(nums,4)
print(test)

def coverint(s):
    s.strip()
    max, min, com = 2 ** 31 - 1, -2 ** 31, 2 ** 31 / 10
    res, flag, i = 0, 1, 1
    if str[0] == "-": flag = -1
    if str[0] == "+": flag = 1
    for c in s[i:]:
        if not '0' <= c <= '9':
            break
        if res > max or res > com and c > "7":
            if flag == 1:
                return max
            else:
                return min
        res += 10 * res + ord(c) - ord('0')
    return res

# def delete1ispalina(s):
#     n = len(s)
#     if n == 1:
#         return s
#     res = []
#     dp = [[False for _ in range(n)] for _ in range(n)]
#     max_length = 0
#     for i in range(n):
#         dp[i][i] = True
#     for j in range(1,n):
#         for i in range(j):
#             if s[i] == s[j]:
#                 if j - i +1 < 3:
#                     dp[i][j] = True
#                 else:
#                     dp[i][j] = dp[i+1][j-1]
#             if dp[i][j] and j-i+1 > max_length:
#                 max_length = j - i + 1
#                 res =s[i:j+1]
#
#     return res
#
# def maxsub(s):
#     n = len(s)
#     if n <= 1:
#         return n
#     dic = {}
#     i,res = -1,0
#     for j in range(n):
#         if s[j] in dic:
#            i = max(i,dic[s[j]])
#         dic[s[j]] = j
#         res = max(res,j-i)
#     return res
#
# def reverseolist(s):
#     s = list(s)
#     n = len(s)
#     if n <= 1 :
#         return s
#     dic = {'a','e','i','o','u'}
#     i,j = 0,n-1
#     while i < j:
#         if s[i] in dic and s[j] in dic:
#             s[i],s[j] = s[j], s[i]
#             i += 1
#             j -= 1
#         elif s[i]  not in dic:
#             i += 1
#     return "".join(s)
#
# def maxlengthpreflix(nums):
#     n = len(nums)
#     if n <= 1:
#         return ''
#     a,b = min(nums),max(nums)
#     for i in range(len(a)):
#         if a[i] != b[i]:
#             break
#     return a[:i]

# def minsum(k):
#     if k > 45: return -1
#
#     def dfs(depth, k, res, path):
#         if k == 0:
#             res.append(int(''.join(path[:])))
#             return res
#         for i in range(depth, 10):
#             if i <= k:
#                 path.append(str(i))
#                 dfs(i + 1, k - i, res, path)
#
#                 path.pop()
#
#     res = []
#     dfs(1, k, res, [])
#     return min(res)

# def minsubarray(nums, k):
#     i, res, tmp = 0, 1, 0
#     for j in range(len(nums)):
#         res *= nums[j]
#         while res >= k:
#             res /= nums[i]
#             i += 1
#         if res < k:
#             tmp += j - i + 1
#     return tmp

def subapple(nums):
    i,res = 0,0
    dic = {}
    for j in range(len(nums)):
        if nums[j] in dic:
            dic[nums[j]] += 1

        dic[nums[j]] = 1



