# Python下午题专项训练

## 考试题型分析

### 下午题特点
- **考试时间**：150分钟
- **题目数量**：通常5-6道大题
- **编程语言**：可选择Python、C、Java等
- **分值分布**：每题15-20分
- **考查重点**：算法实现、代码填空、程序分析

### Python优势
1. **语法简洁**：代码量少，易于实现
2. **内置数据结构**：list、dict、set等功能强大
3. **丰富的库函数**：减少重复编码
4. **易于调试**：逻辑清晰，便于检查

---

## 必考题型专项训练

### 1. 二叉树遍历专项

#### 题型1：基础遍历实现
```python
# 【例题1】实现二叉树的三种遍历方式

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def preorder_traversal(root):
    """前序遍历：根-左-右"""
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # 注意：先压右子树，再压左子树
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    
    return result

def inorder_traversal(root):
    """中序遍历：左-根-右"""
    result = []
    stack = []
    current = root
    
    while stack or current:
        # 一直向左走
        while current:
            stack.append(current)
            current = current.left
        
        # 处理当前节点
        current = stack.pop()
        result.append(current.val)
        
        # 转向右子树
        current = current.right
    
    return result

def postorder_traversal(root):
    """后序遍历：左-右-根"""
    if not root:
        return []
    
    result = []
    stack = [root]
    
    while stack:
        node = stack.pop()
        result.append(node.val)
        
        # 注意：先压左子树，再压右子树
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    
    # 后序遍历需要反转结果
    return result[::-1]

def level_order_traversal(root):
    """层序遍历：逐层从左到右"""
    if not root:
        return []
    
    result = []
    queue = [root]
    
    while queue:
        level_size = len(queue)
        level_nodes = []
        
        for _ in range(level_size):
            node = queue.pop(0)
            level_nodes.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level_nodes)
    
    return result

# 测试用例构建
def build_test_tree():
    """构建测试二叉树
         1
       /   \
      2     3
     / \   /
    4   5 6
    """
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    root.right.left = TreeNode(6)
    return root

# 测试
root = build_test_tree()
print("前序遍历:", preorder_traversal(root))    # [1, 2, 4, 5, 3, 6]
print("中序遍历:", inorder_traversal(root))     # [4, 2, 5, 1, 6, 3]
print("后序遍历:", postorder_traversal(root))   # [4, 5, 2, 6, 3, 1]
print("层序遍历:", level_order_traversal(root)) # [[1], [2, 3], [4, 5, 6]]
```

#### 题型2：二叉树路径问题
```python
# 【例题2】二叉树的所有路径

def binary_tree_paths(root):
    """返回所有从根到叶子的路径"""
    if not root:
        return []
    
    paths = []
    
    def dfs(node, current_path):
        if not node:
            return
        
        # 添加当前节点到路径
        current_path.append(str(node.val))
        
        # 如果是叶子节点，保存路径
        if not node.left and not node.right:
            paths.append("->".join(current_path))
        else:
            # 递归遍历子树
            dfs(node.left, current_path)
            dfs(node.right, current_path)
        
        # 回溯：移除当前节点
        current_path.pop()
    
    dfs(root, [])
    return paths

def path_sum(root, target_sum):
    """找到所有路径和等于目标值的路径"""
    if not root:
        return []
    
    result = []
    
    def dfs(node, current_path, current_sum):
        if not node:
            return
        
        current_path.append(node.val)
        current_sum += node.val
        
        # 叶子节点且路径和等于目标值
        if not node.left and not node.right and current_sum == target_sum:
            result.append(current_path[:])  # 复制当前路径
        
        # 递归遍历子树
        dfs(node.left, current_path, current_sum)
        dfs(node.right, current_path, current_sum)
        
        # 回溯
        current_path.pop()
    
    dfs(root, [], 0)
    return result

# 测试
root = build_test_tree()
print("所有路径:", binary_tree_paths(root))
# 输出: ['1->2->4', '1->2->5', '1->3->6']

# 构建路径和测试树
def build_sum_tree():
    """
         5
       /   \
      4     8
     /     / \
    11    13  4
   / \       / \
  7   2     5   1
    """
    root = TreeNode(5)
    root.left = TreeNode(4)
    root.right = TreeNode(8)
    root.left.left = TreeNode(11)
    root.right.left = TreeNode(13)
    root.right.right = TreeNode(4)
    root.left.left.left = TreeNode(7)
    root.left.left.right = TreeNode(2)
    root.right.right.left = TreeNode(5)
    root.right.right.right = TreeNode(1)
    return root

sum_root = build_sum_tree()
print("路径和为22的路径:", path_sum(sum_root, 22))
# 输出: [[5, 4, 11, 2], [5, 8, 4, 5]]
```

### 2. 图搜索专项

#### 题型1：图的表示和基本遍历
```python
# 【例题3】图的DFS和BFS实现

class Graph:
    def __init__(self):
        self.graph = {}
    
    def add_edge(self, u, v):
        """添加边（无向图）"""
        if u not in self.graph:
            self.graph[u] = []
        if v not in self.graph:
            self.graph[v] = []
        
        self.graph[u].append(v)
        self.graph[v].append(u)
    
    def dfs(self, start):
        """深度优先搜索"""
        visited = set()
        path = []
        
        def dfs_helper(node):
            visited.add(node)
            path.append(node)
            
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    dfs_helper(neighbor)
        
        dfs_helper(start)
        return path
    
    def bfs(self, start):
        """广度优先搜索"""
        visited = set()
        queue = [start]
        visited.add(start)
        path = []
        
        while queue:
            node = queue.pop(0)
            path.append(node)
            
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return path
    
    def is_connected(self):
        """判断图是否连通"""
        if not self.graph:
            return True
        
        start_node = next(iter(self.graph))
        visited_nodes = set(self.dfs(start_node))
        
        return len(visited_nodes) == len(self.graph)
    
    def shortest_path(self, start, end):
        """使用BFS找最短路径"""
        if start == end:
            return [start]
        
        visited = set()
        queue = [(start, [start])]
        visited.add(start)
        
        while queue:
            node, path = queue.pop(0)
            
            for neighbor in self.graph.get(node, []):
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None  # 无路径

# 测试图搜索
g = Graph()
edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
for u, v in edges:
    g.add_edge(u, v)

print("DFS遍历:", g.dfs(0))
print("BFS遍历:", g.bfs(0))
print("图是否连通:", g.is_connected())
print("0到4的最短路径:", g.shortest_path(0, 4))
```

#### 题型2：图的应用问题
```python
# 【例题4】岛屿数量问题

def num_islands(grid):
    """计算岛屿数量（DFS解法）"""
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(i, j):
        # 边界检查和访问检查
        if (i < 0 or i >= rows or j < 0 or j >= cols or 
            grid[i][j] == '0'):
            return
        
        # 标记为已访问
        grid[i][j] = '0'
        
        # 递归访问四个方向
        dfs(i + 1, j)  # 下
        dfs(i - 1, j)  # 上
        dfs(i, j + 1)  # 右
        dfs(i, j - 1)  # 左
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)  # 将整个岛屿标记为已访问
    
    return count

def num_islands_bfs(grid):
    """计算岛屿数量（BFS解法）"""
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def bfs(start_i, start_j):
        queue = [(start_i, start_j)]
        grid[start_i][start_j] = '0'
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            i, j = queue.pop(0)
            
            for di, dj in directions:
                ni, nj = i + di, j + dj
                
                if (0 <= ni < rows and 0 <= nj < cols and 
                    grid[ni][nj] == '1'):
                    grid[ni][nj] = '0'
                    queue.append((ni, nj))
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '1':
                count += 1
                bfs(i, j)
    
    return count

# 测试岛屿问题
grid1 = [
    ["1","1","1","1","0"],
    ["1","1","0","1","0"],
    ["1","1","0","0","0"],
    ["0","0","0","0","0"]
]

grid2 = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"]
]

print("岛屿数量1:", num_islands([row[:] for row in grid1]))  # 1
print("岛屿数量2:", num_islands([row[:] for row in grid2]))  # 3
```

### 3. 动态规划专项

#### 题型1：经典DP问题
```python
# 【例题5】背包问题系列

def knapsack_01(weights, values, capacity):
    """0-1背包问题"""
    n = len(weights)
    # dp[i][w] 表示前i个物品，容量为w的最大价值
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # 不选第i个物品
            dp[i][w] = dp[i-1][w]
            
            # 选第i个物品（如果容量允许）
            if w >= weights[i-1]:
                dp[i][w] = max(dp[i][w], 
                              dp[i-1][w-weights[i-1]] + values[i-1])
    
    return dp[n][capacity]

def knapsack_01_optimized(weights, values, capacity):
    """0-1背包问题（空间优化）"""
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        # 逆序遍历，避免重复使用
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

def knapsack_complete(weights, values, capacity):
    """完全背包问题（物品可重复使用）"""
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        # 正序遍历，允许重复使用
        for w in range(weights[i], capacity + 1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

def knapsack_with_items(weights, values, capacity):
    """返回背包问题的具体方案"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # 填充DP表
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]
            if w >= weights[i-1]:
                dp[i][w] = max(dp[i][w], 
                              dp[i-1][w-weights[i-1]] + values[i-1])
    
    # 回溯找到选择的物品
    selected_items = []
    i, w = n, capacity
    
    while i > 0 and w > 0:
        if dp[i][w] != dp[i-1][w]:
            selected_items.append(i-1)  # 选择了第i-1个物品
            w -= weights[i-1]
        i -= 1
    
    return dp[n][capacity], selected_items[::-1]

# 测试背包问题
weights = [2, 1, 3, 2]
values = [12, 10, 20, 15]
capacity = 5

print("0-1背包最大价值:", knapsack_01(weights, values, capacity))
print("0-1背包优化版:", knapsack_01_optimized(weights, values, capacity))
print("完全背包最大价值:", knapsack_complete(weights, values, capacity))

max_value, items = knapsack_with_items(weights, values, capacity)
print(f"最大价值: {max_value}, 选择物品: {items}")
```

#### 题型2：序列DP问题
```python
# 【例题6】最长公共子序列和最长递增子序列

def longest_common_subsequence(text1, text2):
    """最长公共子序列长度"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def lcs_with_sequence(text1, text2):
    """返回最长公共子序列的具体序列"""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # 回溯构造LCS
    lcs = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return dp[m][n], ''.join(reversed(lcs))

def longest_increasing_subsequence(nums):
    """最长递增子序列长度"""
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # dp[i]表示以nums[i]结尾的LIS长度
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

def lis_with_sequence(nums):
    """返回最长递增子序列的具体序列"""
    if not nums:
        return 0, []
    
    n = len(nums)
    dp = [1] * n
    parent = [-1] * n  # 记录前驱节点
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    
    # 找到最长序列的结束位置
    max_length = max(dp)
    max_index = dp.index(max_length)
    
    # 回溯构造序列
    lis = []
    current = max_index
    
    while current != -1:
        lis.append(nums[current])
        current = parent[current]
    
    return max_length, lis[::-1]

def lis_binary_search(nums):
    """最长递增子序列（二分查找优化，O(n log n)）"""
    if not nums:
        return 0
    
    tails = []  # tails[i]表示长度为i+1的递增子序列的最小尾部
    
    for num in nums:
        # 二分查找插入位置
        left, right = 0, len(tails)
        
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        # 如果num比所有元素都大，添加到末尾
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    
    return len(tails)

# 测试序列DP
text1, text2 = "ABCDGH", "AEDFHR"
print("LCS长度:", longest_common_subsequence(text1, text2))

length, sequence = lcs_with_sequence(text1, text2)
print(f"LCS: 长度={length}, 序列='{sequence}'")

nums = [10, 9, 2, 5, 3, 7, 101, 18]
print("LIS长度:", longest_increasing_subsequence(nums))

lis_len, lis_seq = lis_with_sequence(nums)
print(f"LIS: 长度={lis_len}, 序列={lis_seq}")

print("LIS长度(优化):", lis_binary_search(nums))
```

### 4. 字符串处理专项

#### 题型1：字符串匹配
```python
# 【例题7】字符串匹配算法

def naive_string_match(text, pattern):
    """朴素字符串匹配"""
    n, m = len(text), len(pattern)
    positions = []
    
    for i in range(n - m + 1):
        j = 0
        while j < m and text[i + j] == pattern[j]:
            j += 1
        
        if j == m:
            positions.append(i)
    
    return positions

def kmp_search(text, pattern):
    """KMP字符串匹配算法"""
    def compute_lps_array(pattern):
        """计算最长前缀后缀数组"""
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        
        while i < m:
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    n, m = len(text), len(pattern)
    if m == 0:
        return []
    
    lps = compute_lps_array(pattern)
    positions = []
    
    i = j = 0
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            positions.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    
    return positions

def rabin_karp_search(text, pattern):
    """Rabin-Karp字符串匹配（滚动哈希）"""
    n, m = len(text), len(pattern)
    if m > n:
        return []
    
    # 哈希参数
    base = 256
    prime = 101
    
    # 计算模式串的哈希值
    pattern_hash = 0
    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % prime
    
    # 计算第一个窗口的哈希值
    text_hash = 0
    for i in range(m):
        text_hash = (text_hash * base + ord(text[i])) % prime
    
    # 计算base^(m-1) % prime
    h = 1
    for i in range(m - 1):
        h = (h * base) % prime
    
    positions = []
    
    # 滑动窗口
    for i in range(n - m + 1):
        # 如果哈希值匹配，再逐字符比较
        if pattern_hash == text_hash:
            if text[i:i+m] == pattern:
                positions.append(i)
        
        # 计算下一个窗口的哈希值
        if i < n - m:
            text_hash = (base * (text_hash - ord(text[i]) * h) + 
                        ord(text[i + m])) % prime
            
            # 处理负数
            if text_hash < 0:
                text_hash += prime
    
    return positions

# 测试字符串匹配
text = "ABABDABACDABABCABCABCABCABC"
pattern = "ABABCABCABCABC"

print("朴素匹配:", naive_string_match(text, pattern))
print("KMP匹配:", kmp_search(text, pattern))
print("Rabin-Karp匹配:", rabin_karp_search(text, pattern))
```

#### 题型2：回文字符串
```python
# 【例题8】回文字符串问题

def is_palindrome(s):
    """判断字符串是否为回文"""
    # 方法1：双指针
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    
    return True

def longest_palindrome_substring(s):
    """最长回文子串（中心扩展法）"""
    if not s:
        return ""
    
    def expand_around_center(left, right):
        while (left >= 0 and right < len(s) and 
               s[left] == s[right]):
            left -= 1
            right += 1
        return right - left - 1
    
    start = 0
    max_len = 0
    
    for i in range(len(s)):
        # 奇数长度回文
        len1 = expand_around_center(i, i)
        # 偶数长度回文
        len2 = expand_around_center(i, i + 1)
        
        current_max = max(len1, len2)
        if current_max > max_len:
            max_len = current_max
            start = i - (current_max - 1) // 2
    
    return s[start:start + max_len]

def longest_palindrome_dp(s):
    """最长回文子串（动态规划）"""
    n = len(s)
    if n == 0:
        return ""
    
    # dp[i][j] 表示s[i:j+1]是否为回文
    dp = [[False] * n for _ in range(n)]
    start = 0
    max_len = 1
    
    # 单个字符都是回文
    for i in range(n):
        dp[i][i] = True
    
    # 检查长度为2的子串
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2
    
    # 检查长度大于2的子串
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length
    
    return s[start:start + max_len]

def palindrome_partitioning(s):
    """回文分割：将字符串分割成回文子串"""
    result = []
    
    def is_palindrome_helper(start, end):
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True
    
    def backtrack(start, current_partition):
        if start == len(s):
            result.append(current_partition[:])
            return
        
        for end in range(start, len(s)):
            if is_palindrome_helper(start, end):
                current_partition.append(s[start:end + 1])
                backtrack(end + 1, current_partition)
                current_partition.pop()
    
    backtrack(0, [])
    return result

def count_palindromic_substrings(s):
    """统计回文子串数量"""
    def expand_around_center(left, right):
        count = 0
        while (left >= 0 and right < len(s) and 
               s[left] == s[right]):
            count += 1
            left -= 1
            right += 1
        return count
    
    total_count = 0
    
    for i in range(len(s)):
        # 奇数长度回文
        total_count += expand_around_center(i, i)
        # 偶数长度回文
        total_count += expand_around_center(i, i + 1)
    
    return total_count

# 测试回文问题
test_string = "babad"
print(f"'{test_string}' 是否回文:", is_palindrome(test_string))
print("最长回文子串(中心扩展):", longest_palindrome_substring(test_string))
print("最长回文子串(DP):", longest_palindrome_dp(test_string))

partition_string = "aab"
print(f"'{partition_string}' 的回文分割:", palindrome_partitioning(partition_string))

count_string = "abc"
print(f"'{count_string}' 的回文子串数量:", count_palindromic_substrings(count_string))
```

### 5. 递归专项

#### 题型1：经典递归问题
```python
# 【例题9】递归经典问题

def hanoi_towers(n, source, destination, auxiliary):
    """汉诺塔问题"""
    moves = []
    
    def hanoi_helper(n, src, dest, aux):
        if n == 1:
            moves.append(f"移动盘子 {n} 从 {src} 到 {dest}")
            return
        
        # 将前n-1个盘子从source移到auxiliary
        hanoi_helper(n - 1, src, aux, dest)
        
        # 将第n个盘子从source移到destination
        moves.append(f"移动盘子 {n} 从 {src} 到 {dest}")
        
        # 将前n-1个盘子从auxiliary移到destination
        hanoi_helper(n - 1, aux, dest, src)
    
    hanoi_helper(n, source, destination, auxiliary)
    return moves

def generate_parentheses(n):
    """生成所有有效的括号组合"""
    result = []
    
    def backtrack(current, open_count, close_count):
        # 基础情况：生成了n对括号
        if len(current) == 2 * n:
            result.append(current)
            return
        
        # 添加左括号（如果还有剩余）
        if open_count < n:
            backtrack(current + "(", open_count + 1, close_count)
        
        # 添加右括号（如果右括号数量少于左括号）
        if close_count < open_count:
            backtrack(current + ")", open_count, close_count + 1)
    
    backtrack("", 0, 0)
    return result

def permutations(nums):
    """生成全排列"""
    result = []
    
    def backtrack(current_permutation, remaining):
        if not remaining:
            result.append(current_permutation[:])
            return
        
        for i in range(len(remaining)):
            # 选择
            current_permutation.append(remaining[i])
            # 递归
            backtrack(current_permutation, remaining[:i] + remaining[i+1:])
            # 撤销选择
            current_permutation.pop()
    
    backtrack([], nums)
    return result

def combinations(n, k):
    """生成组合C(n,k)"""
    result = []
    
    def backtrack(start, current_combination):
        if len(current_combination) == k:
            result.append(current_combination[:])
            return
        
        for i in range(start, n + 1):
            current_combination.append(i)
            backtrack(i + 1, current_combination)
            current_combination.pop()
    
    backtrack(1, [])
    return result

def subsets(nums):
    """生成所有子集"""
    result = []
    
    def backtrack(start, current_subset):
        result.append(current_subset[:])
        
        for i in range(start, len(nums)):
            current_subset.append(nums[i])
            backtrack(i + 1, current_subset)
            current_subset.pop()
    
    backtrack(0, [])
    return result

# 测试递归问题
print("汉诺塔(3个盘子):")
hanoi_moves = hanoi_towers(3, 'A', 'C', 'B')
for move in hanoi_moves:
    print(move)

print("\n生成3对括号的所有组合:")
print(generate_parentheses(3))

print("\n[1,2,3]的全排列:")
print(permutations([1, 2, 3]))

print("\nC(4,2)的所有组合:")
print(combinations(4, 2))

print("\n[1,2,3]的所有子集:")
print(subsets([1, 2, 3]))
```

### 6. 排序与查找专项

#### 题型1：高级排序算法
```python
# 【例题10】排序算法实现

def quick_sort(arr):
    """快速排序"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

def quick_sort_inplace(arr, low=0, high=None):
    """原地快速排序"""
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        # 分区操作
        pivot_index = partition(arr, low, high)
        
        # 递归排序左右子数组
        quick_sort_inplace(arr, low, pivot_index - 1)
        quick_sort_inplace(arr, pivot_index + 1, high)

def partition(arr, low, high):
    """分区函数"""
    pivot = arr[high]  # 选择最后一个元素作为基准
    i = low - 1  # 小于基准的元素的索引
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def merge_sort(arr):
    """归并排序"""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    """合并两个有序数组"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def heap_sort(arr):
    """堆排序"""
    def heapify(arr, n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2
        
        if left < n and arr[left] > arr[largest]:
            largest = left
        
        if right < n and arr[right] > arr[largest]:
            largest = right
        
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)
    
    n = len(arr)
    
    # 构建最大堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    # 逐个提取元素
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    
    return arr

# 测试排序算法
test_array = [64, 34, 25, 12, 22, 11, 90]
print("原数组:", test_array)
print("快速排序:", quick_sort(test_array.copy()))
print("归并排序:", merge_sort(test_array.copy()))
print("堆排序:", heap_sort(test_array.copy()))

# 原地快速排序测试
inplace_array = test_array.copy()
quick_sort_inplace(inplace_array)
print("原地快速排序:", inplace_array)
```

#### 题型2：查找算法变种
```python
# 【例题11】查找算法及其变种

def binary_search_basic(arr, target):
    """基础二分查找"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def binary_search_first_occurrence(arr, target):
    """查找第一个出现的位置"""
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            result = mid
            right = mid - 1  # 继续向左查找
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def binary_search_last_occurrence(arr, target):
    """查找最后一个出现的位置"""
    left, right = 0, len(arr) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            result = mid
            left = mid + 1  # 继续向右查找
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def search_insert_position(arr, target):
    """查找插入位置"""
    left, right = 0, len(arr)
    
    while left < right:
        mid = (left + right) // 2
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left

def search_in_rotated_sorted_array(arr, target):
    """在旋转排序数组中查找"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        
        # 判断哪一半是有序的
        if arr[left] <= arr[mid]:  # 左半部分有序
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # 右半部分有序
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

def find_peak_element(arr):
    """查找峰值元素"""
    left, right = 0, len(arr) - 1
    
    while left < right:
        mid = (left + right) // 2
        
        if arr[mid] > arr[mid + 1]:
            right = mid
        else:
            left = mid + 1
    
    return left

def sqrt_binary_search(x):
    """使用二分查找计算平方根"""
    if x < 2:
        return x
    
    left, right = 1, x // 2
    
    while left <= right:
        mid = (left + right) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # 返回最大的整数平方根

# 测试查找算法
sorted_array = [1, 2, 2, 2, 3, 4, 5]
target = 2

print("基础二分查找:", binary_search_basic(sorted_array, target))
print("第一个出现位置:", binary_search_first_occurrence(sorted_array, target))
print("最后一个出现位置:", binary_search_last_occurrence(sorted_array, target))
print("插入位置:", search_insert_position(sorted_array, target))

rotated_array = [4, 5, 6, 7, 0, 1, 2]
print("旋转数组中查找0:", search_in_rotated_sorted_array(rotated_array, 0))

peak_array = [1, 2, 3, 1]
print("峰值元素位置:", find_peak_element(peak_array))

print("25的平方根:", sqrt_binary_search(25))
```

---

## 考试答题策略

### 1. 时间分配策略
```
总时间：150分钟
- 阅读理解：15分钟
- 二叉树遍历：25分钟
- 图搜索算法：25分钟
- 动态规划：30分钟
- 字符串处理：20分钟
- 其他算法：20分钟
- 检查验证：15分钟
```

### 2. 答题顺序建议
1. **先做熟悉的题型**：从最有把握的开始
2. **高分题优先**：优先完成分值高的题目
3. **预留检查时间**：确保代码正确性

### 3. Python代码规范
```python
# 标准函数模板
def function_name(parameters):
    """
    函数功能描述
    参数：parameter - 参数说明
    返回：返回值说明
    时间复杂度：O(n)
    空间复杂度：O(1)
    """
    # 边界条件处理
    if not parameters:
        return default_value
    
    # 主要算法逻辑
    result = algorithm_implementation(parameters)
    
    return result

# 测试用例
if __name__ == "__main__":
    test_cases = [
        # (输入, 期望输出)
        ([1, 2, 3], expected_output),
    ]
    
    for input_data, expected in test_cases:
        result = function_name(input_data)
        print(f"输入: {input_data}, 输出: {result}, 期望: {expected}")
        assert result == expected, f"测试失败: {result} != {expected}"
```

### 4. 常见错误避免
1. **边界条件**：空数组、单元素、负数等
2. **索引越界**：数组访问时注意边界
3. **递归终止**：确保递归有正确的终止条件
4. **变量初始化**：确保变量正确初始化
5. **算法复杂度**：注意时间和空间复杂度要求

### 5. 调试技巧
```python
# 添加调试输出
def debug_function(arr):
    print(f"输入数组: {arr}")
    
    # 关键步骤输出
    for i, val in enumerate(arr):
        print(f"处理第{i}个元素: {val}")
    
    result = process(arr)
    print(f"最终结果: {result}")
    
    return result

# 使用断言验证
def validate_result(input_data, result):
    assert isinstance(result, expected_type), "返回类型错误"
    assert len(result) == expected_length, "返回长度错误"
    # 其他验证逻辑
```

通过这些专项训练和策略，你可以系统地提高Python编程题的解题能力，在考试中取得好成绩。记住多练习、多总结，熟练掌握各种算法模板。