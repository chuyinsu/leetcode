import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

public class Solution {
	class ListNode {
		int val;
		ListNode next;

		ListNode(int x) {
			val = x;
			next = null;
		}
	}

	public class TreeLinkNode {
		int val;
		TreeLinkNode left, right, next;

		TreeLinkNode(int x) {
			val = x;
		}
	}

	public class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
		}
	}

	public class Interval {
		int start;
		int end;

		Interval() {
			start = 0;
			end = 0;
		}

		Interval(int s, int e) {
			start = s;
			end = e;
		}
	}

	class RandomListNode {
		int label;
		RandomListNode next, random;

		RandomListNode(int x) {
			this.label = x;
		}
	}

	class UndirectedGraphNode {
		int label;
		ArrayList<UndirectedGraphNode> neighbors;

		UndirectedGraphNode(int x) {
			label = x;
			neighbors = new ArrayList<UndirectedGraphNode>();
		}
	}

	// Single Number
	public int singleNumber(int[] A) {
		if (A == null) {
			return 0;
		}
		int missingNumber = 0;
		for (int i = 0; i < A.length; i++) {
			missingNumber ^= A[i];
		}
		return missingNumber;
	}

	// Binary Gap
	public int binaryGap(int n) {
		int mask = 1;
		int gap = 0;
		while ((mask & n) == 0 && mask != 0) {
			mask <<= 1;
		}
		while (mask != 0) {
			while ((mask & n) != 0) {
				mask <<= 1;
			}
			int counter = 0;
			while ((mask & n) == 0 && mask != 0) {
				mask <<= 1;
				counter++;
			}
			if (mask != 0) {
				gap = Math.max(gap, counter);
			}
		}
		return gap;
	}

	// Word Search
	public boolean exist(char[][] board, String word) {
		if (board == null || word == null) {
			return false;
		}
		boolean[][] memo = new boolean[board.length][board[0].length];
		for (int i = 0; i < memo.length; i++) {
			Arrays.fill(memo[i], false);
		}
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[0].length; j++) {
				if (existHelper(board, word, i, j, memo)) {
					return true;
				}
			}
		}
		return false;
	}

	private boolean existHelper(char[][] board, String word, int row, int col,
			boolean[][] memo) {
		if (word.isEmpty()) {
			return true;
		}
		if (row < 0 || row >= board.length || col < 0 || col >= board[0].length
				|| memo[row][col]) {
			return false;
		}
		if (word.charAt(0) == board[row][col]) {
			String rest = word.substring(1, word.length());
			memo[row][col] = true;
			if (existHelper(board, rest, row + 1, col, memo)
					|| existHelper(board, rest, row - 1, col, memo)
					|| existHelper(board, rest, row, col + 1, memo)
					|| existHelper(board, rest, row, col - 1, memo)) {
				return true;
			} else {
				memo[row][col] = false;
			}
		}
		return false;
	}

	// Linked List Cycle II
	public ListNode detectCycle(ListNode head) {
		if (head == null || head.next == null) {
			return null;
		}
		ListNode walker = head.next;
		ListNode runner = head.next.next;
		while (runner != null && runner.next != null && walker != runner) {
			walker = walker.next;
			runner = runner.next.next;
		}
		if (runner == null || runner.next == null) {
			return null;
		}
		walker = head;
		while (walker != runner) {
			walker = walker.next;
			runner = runner.next;
		}
		return walker;
	}

	// Remove Element
	public int removeElement(int[] A, int elem) {
		if (A == null) {
			return 0;
		}
		int scan = 0;
		int slot = 0;
		while (scan < A.length) {
			if (A[scan] != elem) {
				A[slot++] = A[scan];
			}
			scan++;
		}
		return slot;
	}

	// Pow(x, n)
	public double pow(double x, int n) {
		if (n == 0) {
			return 1;
		}
		if (n == 1) {
			return x;
		}
		double half = pow(x, Math.abs(n / 2));
		double result = half * half;
		if (n % 2 != 0) {
			result *= x;
		}
		if (n < 0) {
			result = 1 / result;
		}
		return result;
	}

	// Largest Rectangle in Histogram
	public int largestRectangleArea(int[] height) {
		if (height == null) {
			return 0;
		}
		int result = 0;
		int outMost = 0;
		int valIndex = 0;
		int length = 0;
		int index = 0;
		Stack<Integer> stack = new Stack<Integer>();
		while (index < height.length) {
			if (stack.isEmpty() || height[index] >= height[stack.peek()]) {
				stack.push(index++);
			} else {
				outMost = stack.peek();
				do {
					valIndex = stack.pop();
					length = stack.isEmpty() ? outMost + 1 : outMost
							- stack.peek();
					result = Math.max(result, height[valIndex] * length);
				} while (!stack.isEmpty()
						&& height[index] < height[stack.peek()]);
				stack.push(index++);
			}
		}
		if (!stack.isEmpty()) {
			outMost = stack.peek();
		}
		while (!stack.isEmpty()) {
			valIndex = stack.pop();
			length = stack.isEmpty() ? outMost + 1 : outMost - stack.peek();
			result = Math.max(result, height[valIndex] * length);
		}
		return result;
	}

	// Trapping Rain Water
	public int trap(int[] A) {
		if (A == null || A.length < 3) {
			return 0;
		}
		int[] maxLeft = new int[A.length];
		int[] maxRight = new int[A.length];
		Arrays.fill(maxLeft, 0);
		Arrays.fill(maxRight, 0);
		maxLeft[0] = A[0];
		for (int i = 1; i < A.length; i++) {
			maxLeft[i] = Math.max(A[i], maxLeft[i - 1]);
		}
		maxRight[maxRight.length - 1] = A[A.length - 1];
		for (int i = maxRight.length - 2; i >= 0; i--) {
			maxRight[i] = Math.max(A[i], maxRight[i + 1]);
		}
		int result = 0;
		for (int i = 1; i < A.length - 1; i++) {
			int min = Math.min(maxLeft[i - 1], maxRight[i + 1]);
			result += Math.max(0, min - A[i]);
		}
		return result;
	}

	// Reorder List
	public void reorderList(ListNode head) {
		if (head == null || head.next == null || head.next.next == null) {
			return;
		}
		ListNode walker = head.next;
		ListNode prevWalker = head;
		ListNode runner = walker.next;
		ListNode prevRunner = walker;
		while (runner != null && runner.next != null) {
			prevWalker = walker;
			walker = walker.next;
			runner = runner.next;
			prevRunner = runner;
			runner = runner.next;
		}
		runner = runner == null ? prevRunner : runner;

		// reverse second half of the list
		reverse(walker, runner);

		prevWalker.next = null;
		ListNode l1 = head;
		ListNode l2 = runner;
		ListNode r1 = l1.next;
		ListNode r2 = l2.next;
		while (l1 != null && l2 != null) {
			l1.next = l2;
			l2.next = r1 == null ? l2.next : r1;
			l1 = r1;
			l2 = r2;
			r1 = r1 == null ? null : r1.next;
			r2 = r2 == null ? null : r2.next;
		}
	}

	private void reverse(ListNode head, ListNode tail) {
		if (head == tail) {
			return;
		}
		ListNode prev = head;
		ListNode curr = prev.next;
		ListNode post = curr.next;
		while (curr != null) {
			curr.next = prev;
			prev = curr;
			curr = post;
			post = post == null ? null : post.next;
		}
		head.next = null;
	}

	// Single Number II
	public int singleNumberII(int[] A) {
		if (A == null) {
			return 0;
		}
		int ones = A[0];
		int twos = 0;
		int threes = 0;
		for (int i = 1; i < A.length; i++) {
			twos |= (A[i] & ones);
			ones ^= A[i];
			threes = ones & twos;
			ones &= (~threes);
			twos &= (~threes);
		}
		return ones;
	}

	// N-Queens
	public ArrayList<String[]> solveNQueens(int n) {
		ArrayList<String[]> result = new ArrayList<String[]>();
		if (n < 1) {
			return result;
		}
		int[] solution = new int[n];
		solveNQueensHelper(solution, 0, result);
		return result;
	}

	private void solveNQueensHelper(int[] solution, int row,
			ArrayList<String[]> result) {
		if (row >= solution.length) {
			String[] board = new String[solution.length];
			for (int i = 0; i < solution.length; i++) {
				char[] boardRow = new char[solution.length];
				Arrays.fill(boardRow, '.');
				boardRow[solution[i]] = 'Q';
				board[i] = new String(boardRow);
			}
			result.add(board);
			return;
		}
		for (int i = 0; i < solution.length; i++) {
			solution[row] = i;
			if (valid(solution, row)) {
				solveNQueensHelper(solution, row + 1, result);
			}
		}
	}

	private boolean valid(int[] solution, int row) {
		for (int i = 0; i < row; i++) {
			if ((solution[i] == solution[row])
					|| (solution[i] - i == solution[row] - row)
					|| (solution[i] + i == solution[row] + row)) {
				return false;
			}
		}
		return true;
	}

	// Populating Next Right Pointers in Each Node II
	public void connectII(TreeLinkNode root) {
		if (root == null) {
			return;
		}
		root.next = null;
		TreeLinkNode begin = null;
		TreeLinkNode end = null;
		TreeLinkNode prev = root;
		while (prev != null) {
			while (prev != null) {
				while (prev != null && prev.left == null && prev.right == null) {
					prev = prev.next;
				}
				if (prev != null) {
					TreeLinkNode[] nodes = process(prev.left, begin, end);
					begin = nodes[0];
					end = nodes[1];
					nodes = process(prev.right, begin, end);
					begin = nodes[0];
					end = nodes[1];
					prev = prev.next;
				}
			}
			prev = begin;
			begin = null;
			end = null;
		}
	}

	private TreeLinkNode[] process(TreeLinkNode node, TreeLinkNode begin,
			TreeLinkNode end) {
		if (node == null) {
			return new TreeLinkNode[] { begin, end };
		} else if (end == null) {
			end = node;
			begin = node;
		} else {
			end.next = node;
			end = node;
		}
		return new TreeLinkNode[] { begin, end };
	}

	// Best Time to Buy and Sell Stock III
	public int maxProfitIII(int[] prices) {
		if (prices == null || prices.length == 0) {
			return 0;
		}
		int[] leftProfit = new int[prices.length];
		int[] rightProfit = new int[prices.length];
		int min = prices[0];
		Arrays.fill(leftProfit, 0);
		for (int i = 1; i < leftProfit.length; i++) {
			min = Math.min(min, prices[i]);
			leftProfit[i] = Math.max(leftProfit[i],
					Math.max(leftProfit[i - 1], prices[i] - min));
		}
		int max = prices[prices.length - 1];
		Arrays.fill(rightProfit, 0);
		for (int i = rightProfit.length - 2; i >= 0; i--) {
			max = Math.max(max, prices[i]);
			rightProfit[i] = Math.max(rightProfit[i],
					Math.max(rightProfit[i + 1], max - prices[i]));
		}
		int profit = 0;
		for (int i = 0; i < prices.length; i++) {
			profit = Math.max(profit, leftProfit[i] + rightProfit[i]);
		}
		return profit;
	}

	// Convert Sorted List to Binary Search Tree
	public TreeNode sortedListToBST(ListNode head) {
		if (head == null) {
			return null;
		}
		if (head.next == null) {
			return new TreeNode(head.val);
		}
		ListNode prev = null;
		ListNode post = null;
		ListNode walker = head;
		ListNode runner = head;
		while (runner != null && runner.next != null) {
			prev = walker;
			walker = walker.next;
			runner = runner.next.next;
		}
		prev.next = null;
		post = walker.next;
		walker.next = null;
		TreeNode root = new TreeNode(walker.val);
		root.left = sortedListToBST(head);
		root.right = sortedListToBST(post);
		return root;
	}

	// Permutations
	public ArrayList<ArrayList<Integer>> permute(int[] num) {
		if (num == null) {
			return null;
		}
		ArrayDeque<ArrayList<Integer>> queue = new ArrayDeque<ArrayList<Integer>>();
		ArrayList<Integer> empty = new ArrayList<Integer>();
		queue.offer(empty);
		for (int i = 0; i < num.length; i++) {
			int len = queue.size();
			for (int j = 0; j < len; j++) {
				ArrayList<Integer> base = queue.poll();
				for (int k = 0; k <= base.size(); k++) {
					ArrayList<Integer> permutation = new ArrayList<Integer>(
							base);
					permutation.add(k, num[i]);
					queue.offer(permutation);
				}
			}
		}
		return new ArrayList<ArrayList<Integer>>(queue);
	}

	// Merge Sorted Array
	public void merge(int A[], int m, int B[], int n) {
		if (A == null || B == null || m < 0 || n < 0) {
			return;
		}
		int index = m + n - 1;
		int indexA = m - 1;
		int indexB = n - 1;
		while (indexA >= 0 && indexB >= 0) {
			A[index--] = A[indexA] > B[indexB] ? A[indexA--] : B[indexB--];
		}
		while (indexB >= 0) {
			A[index--] = B[indexB--];
		}
	}

	// Median of Two Sorted Arrays
	// Always look at the "right" position
	public double findMedianSortedArraysRight(int A[], int B[]) {
		return findMedianSortedArraysHelper(A, B,
				Math.max(0, (A.length - B.length) / 2),
				Math.min(A.length - 1, (A.length + B.length) / 2));
	}

	private double findMedianSortedArraysHelper(int[] A, int[] B, int left,
			int right) {
		if (left > right) {
			return findMedianSortedArraysHelper(B, A,
					Math.max(0, (B.length - A.length) / 2),
					Math.min(B.length - 1, (B.length + A.length) / 2));
		}
		int mid = left + (right - left) / 2;
		int compare = (A.length + B.length) / 2 - mid;

		if ((compare == 0 || B[compare - 1] <= A[mid])
				&& (compare == B.length || A[mid] <= B[compare])) {
			if ((A.length + B.length) % 2 != 0) {
				return A[mid];
			} else {
				int prevA = mid > 0 ? A[mid - 1] : Integer.MIN_VALUE;
				int prevB = compare > 0 ? B[compare - 1] : Integer.MIN_VALUE;
				return (A[mid] + Math.max(prevA, prevB)) / 2D;
			}
		}
		if (compare < 0
				|| (compare >= 0 && compare < B.length && A[mid] > B[compare])) {
			return findMedianSortedArraysHelper(A, B, left, mid - 1);
		} else {
			return findMedianSortedArraysHelper(A, B, mid + 1, right);
		}
	}

	// Median of Two Sorted Arrays
	// Always look at the "left" position
	public double findMedianSortedArraysLeft(int A[], int B[]) {
		return findMedianSortedArrays(A, B,
				Math.max(0, (A.length - B.length - 1) / 2),
				Math.min(A.length - 1, (A.length + B.length - 1) / 2));
	}

	private double findMedianSortedArrays(int[] A, int[] B, int left, int right) {
		if (left > right) {
			return findMedianSortedArrays(B, A,
					Math.max(0, (B.length - A.length - 1) / 2),
					Math.min(B.length - 1, (A.length + B.length - 1) / 2));
		}
		int mid = left + (right - left) / 2;
		int compare = (A.length + B.length - 1) / 2 - mid;
		if ((compare == 0 || A[mid] >= B[compare - 1])
				&& (compare == B.length || A[mid] <= B[compare])) {
			if ((A.length + B.length) % 2 != 0) {
				return A[mid];
			} else {
				int postA = mid == A.length - 1 ? Integer.MAX_VALUE
						: A[mid + 1];
				int postB = compare == B.length ? Integer.MAX_VALUE
						: B[compare];
				return (A[mid] + Math.min(postA, postB)) / 2D;
			}
		}
		if (compare < 0
				|| (compare >= 0 && compare < B.length && A[mid] > B[compare])) {
			return findMedianSortedArrays(A, B, left, mid - 1);
		} else {
			return findMedianSortedArrays(A, B, mid + 1, right);
		}
	}

	// Two Sum
	public int[] twoSum(int[] numbers, int target) {
		HashMap<Integer, Integer> memo = new HashMap<Integer, Integer>();
		for (int i = 0; i < numbers.length; i++) {
			int remain = target - numbers[i];
			if (memo.containsKey(remain)) {
				return new int[] { memo.get(remain) + 1, i + 1 };
			} else {
				memo.put(numbers[i], i);
			}
		}
		return null;
	}

	// Regular Expression Matching
	public boolean isMatch(String s, String p) {
		if (s == null || p == null) {
			return false;
		}
		return isMatchHelper(s, p, 0, 0);
	}

	private boolean isMatchHelper(String s, String p, int indexS, int indexP) {
		if (indexS == s.length() && indexP == p.length()) {
			return true;
		} else if (indexP == p.length()) {
			return false;
		} else if (indexP < p.length() - 1 && p.charAt(indexP + 1) == '*') {
			int match = 0;
			while (indexS + match < s.length()
					&& (s.charAt(indexS + match) == p.charAt(indexP) || p
							.charAt(indexP) == '.')) {
				match++;
			}
			for (int i = 0; i <= match; i++) {
				if (isMatchHelper(s, p, indexS + i, indexP + 2)) {
					return true;
				}
			}
			return false;
		} else if (indexS == s.length()) {
			return false;
		} else if (s.charAt(indexS) == p.charAt(indexP)
				|| p.charAt(indexP) == '.') {
			return isMatchHelper(s, p, indexS + 1, indexP + 1);
		} else {
			return false;
		}
	}

	// Longest Consecutive Sequence
	public int longestConsecutive(int[] num) {
		if (num == null || num.length == 0) {
			return 0;
		}
		int max = 1;
		HashMap<Integer, Integer> memo = new HashMap<Integer, Integer>();
		for (int i : num) {
			if (!memo.containsKey(i)) {
				int left = memo.containsKey(i - 1) ? i - 1 - memo.get(i - 1)
						+ 1 : i;
				int right = memo.containsKey(i + 1) ? i + 1 + memo.get(i + 1)
						- 1 : i;
				int len = right - left + 1;
				max = Math.max(max, len);
				memo.put(left, len);
				memo.put(right, len);
				if (left != i && right != i) {
					memo.put(i, len);
				}
			}
		}
		return max;
	}

	// Construct Binary Tree from Preorder and Inorder Traversal
	public TreeNode buildTree(int[] preorder, int[] inorder) {
		return buildTreeHelper(preorder, 0, preorder.length - 1, inorder, 0,
				inorder.length - 1);
	}

	private TreeNode buildTreeHelper(int[] preorder, int i1, int j1,
			int inorder[], int i2, int j2) {
		if (i1 > j1 || i2 > j2) {
			return null;
		}
		TreeNode root = new TreeNode(preorder[i1]);
		for (int i = i2; i <= j2; i++) {
			if (inorder[i] == preorder[i1]) {
				root.left = buildTreeHelper(preorder, i1 + 1, i1 + i - i2,
						inorder, i2, i - 1);
				root.right = buildTreeHelper(preorder, i1 + i - i2 + 1, j1,
						inorder, i + 1, j2);
				break;
			}
		}
		return root;
	}

	// Reverse Integer
	public int reverse(int x) {
		int r = 0;
		while (x != 0) {
			r = r * 10 + x % 10;
			x /= 10;
		}
		return r;
	}

	// Reverse Bits
	public int reverseBits(int x) {
		int mask = 1;
		int result = 0;
		for (int i = Integer.SIZE - 1; i >= 0; i--) {
			result |= ((mask & x) << i);
			x >>>= 1;
		}
		return result;
	}

	// Convert Sorted Array to Binary Search Tree
	public TreeNode sortedArrayToBST(int[] num) {
		if (num == null) {
			return null;
		}
		return sortedArrayToBSTHelper(num, 0, num.length - 1);
	}

	private TreeNode sortedArrayToBSTHelper(int[] num, int left, int right) {
		if (left > right) {
			return null;
		}
		int mid = left + (right - left) / 2;
		TreeNode root = new TreeNode(num[mid]);
		root.left = sortedArrayToBSTHelper(num, left, mid - 1);
		root.right = sortedArrayToBSTHelper(num, mid + 1, right);
		return root;
	}

	// Reverse Linked List II
	public ListNode reverseBetween(ListNode head, int m, int n) {
		ListNode start = head;
		ListNode prevStart = null;
		ListNode end = head;
		ListNode postEnd = null;
		for (int i = 0; i < n - m; i++) {
			end = end.next;
		}
		for (int i = 0; i < m - 1; i++) {
			prevStart = start;
			start = start.next;
			end = end.next;
		}
		postEnd = end.next;
		reverseBetweenHelper(start, end);
		start.next = postEnd;
		if (prevStart == null) {
			return end;
		} else {
			prevStart.next = end;
			return head;
		}
	}

	private void reverseBetweenHelper(ListNode start, ListNode end) {
		if (start == end) {
			return;
		}
		ListNode prev = start;
		ListNode curr = prev.next;
		ListNode post = curr.next;
		while (prev != end) {
			curr.next = prev;
			prev = curr;
			curr = post;
			post = post == null ? null : post.next;
		}
	}

	// Search for a Range
	public int[] searchRange(int[] A, int target) {
		return new int[] { searchRangeHelper(A, target, 0, A.length - 1, -1),
				searchRangeHelper(A, target, 0, A.length - 1, 1) };
	}

	private int searchRangeHelper(int[] A, int target, int start, int end,
			int step) {
		if (start > end) {
			return -1;
		}
		int mid = start + (end - start) / 2;
		if (A[mid] < target) {
			return searchRangeHelper(A, target, mid + 1, end, step);
		} else if (A[mid] > target) {
			return searchRangeHelper(A, target, start, mid - 1, step);
		} else {
			if (step < 0) {
				if (mid == 0 || A[mid + step] < target) {
					return mid;
				} else {
					return searchRangeHelper(A, target, start, mid - 1, step);
				}
			} else {
				if (mid == A.length - 1 || A[mid + step] > target) {
					return mid;
				} else {
					return searchRangeHelper(A, target, mid + 1, end, step);
				}
			}
		}
	}

	// Word Break II
	public ArrayList<String> wordBreakII(String s, Set<String> dict) {
		HashMap<String, ArrayList<String>> memo = new HashMap<String, ArrayList<String>>();
		return wordBreakHelper(s, dict, 0, 0, memo);
	}

	private ArrayList<String> wordBreakHelper(String s, Set<String> dict,
			int start, int end, HashMap<String, ArrayList<String>> memo) {
		String key = start + "-" + end;
		if (memo.containsKey(key)) {
			return memo.get(key);
		}
		ArrayList<String> result = new ArrayList<String>();
		String word = new String(s.substring(start, end + 1));
		if (end == s.length() - 1) {
			if (dict.contains(word)) {
				result.add(word);
			}
			return result;
		}
		ArrayList<String> partialResult1 = wordBreakHelper(s, dict, start,
				end + 1, memo);
		if (dict.contains(word)) {
			ArrayList<String> partialResult2 = wordBreakHelper(s, dict,
					end + 1, end + 1, memo);
			for (String string : partialResult2) {
				result.add(word + " " + string);
			}
		}
		result.addAll(partialResult1);
		memo.put(key, result);
		return result;
	}

	// Permutation Sequence
	public String getPermutation(int n, int k) {
		StringBuffer sb = new StringBuffer();
		for (int i = 1; i <= n; i++) {
			sb.append(i);
		}
		int m = 1;
		for (int i = 1; i <= n - 1; i++) {
			m *= i;
		}
		StringBuffer result = new StringBuffer();
		while (sb.length() > 0) {
			int r = (k - 1) / m;
			result.append(sb.charAt(r));
			sb.deleteCharAt(r);
			n--;
			k -= r * m;
			m /= (n == 0 ? 1 : n);
		}
		return result.toString();
	}

	// Construct Binary Tree from Inorder and Postorder Traversal
	public TreeNode buildTreeII(int[] inorder, int[] postorder) {
		if (inorder == null || postorder == null) {
			return null;
		}
		return buildTreeHelperII(inorder, 0, inorder.length - 1, postorder, 0,
				postorder.length - 1);
	}

	private TreeNode buildTreeHelperII(int[] inorder, int i1, int j1,
			int[] postorder, int i2, int j2) {
		if (i1 > j1 || i2 > j2) {
			return null;
		}
		TreeNode root = new TreeNode(postorder[j2]);
		for (int i = i1; i <= j1; i++) {
			if (inorder[i] == postorder[j2]) {
				root.left = buildTreeHelperII(inorder, i1, i - 1, postorder,
						i2, i2 + i - i1 - 1);
				root.right = buildTreeHelperII(inorder, i + 1, j1, postorder,
						i2 + i - i1, j2 - 1);
				break;
			}
		}
		return root;
	}

	// Simple Sorting

	// O(n^2) simple but slow
	public void bubbleSort(int[] data) {
		// move backward from the last index to 1
		for (int out = data.length - 1; out >= 1; out--) {
			// move forward from 0 to the right
			// BUBBLE up the largest value to the right
			for (int in = 0; in < out; in++) {
				if (data[in] > data[in + 1])
					swap(data, in, in + 1);
			}
		}
	}

	// O(n^2) faster than bubble sort because swap only happens in the outer
	// loop
	public void selectionSort(int[] data) {
		int min; // set min variable for tmp min value
		// move forward to right to SELECT the minimum value
		for (int out = 0; out < data.length - 1; out++) {
			min = out; // set initial min index to be out
			// move forward to right from out+1 to the end
			for (int in = out + 1; in < data.length; in++) {
				// if data is smaller than current min value
				if (data[in] < data[min])
					min = in; // set a new min index
			}
			// swap min value with the first one as we move forward to the right
			// swapping is happening in the outer loop
			if (out != min)
				swap(data, out, min);
		}
	}

	// O(n^2) fastest among the three 1. less number of comparisons 2. uses
	// shifting instead of swapping
	public void insertionSort(int[] data) {
		// start from 1 till the last index
		for (int out = 1; out < data.length; out++) {
			int tmp = data[out]; // save the first value as tmp
			int in = out; // initial in variable index
			// move backward till it finds the location to insert
			while (in > 0 && data[in - 1] >= tmp) {
				// shift to right to make a room
				data[in] = data[in - 1];
				in--;
			}
			// finally INSERT the tmp value to the right position
			data[in] = tmp;
		}
	}

	// helper method to swap two values in an array
	private void swap(int[] data, int one, int two) {
		int tmp = data[one];
		data[one] = data[two];
		data[two] = tmp;
	}

	// Remove Duplicates from Sorted Array
	public int removeDuplicates(int[] A) {
		if (A == null) {
			return 0;
		} else if (A.length < 2) {
			return A.length;
		}
		int scan = 1;
		int slot = 1;
		while (scan < A.length) {
			if (A[scan] != A[slot - 1]) {
				A[slot++] = A[scan++];
			} else {
				scan++;
			}
		}
		return slot;
	}

	// Remove Duplicates from Sorted List
	public ListNode deleteDuplicates(ListNode head) {
		ListNode tail = head;
		ListNode scan = null;
		while (tail != null) {
			scan = tail.next;
			while (scan != null && scan.val == tail.val) {
				scan = scan.next;
			}
			tail.next = scan;
			tail = scan;
		}
		return head;
	}

	// Same Tree
	public boolean isSameTree(TreeNode p, TreeNode q) {
		if (p == null && q == null) {
			return true;
		} else if (p == null || q == null) {
			return false;
		} else if (p.val == q.val) {
			return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
		} else {
			return false;
		}
	}

	// Remove Duplicates from Sorted Array II
	public int removeDuplicatesII(int[] A) {
		if (A == null) {
			return 0;
		} else if (A.length < 3) {
			return A.length;
		}
		int scan = 2;
		int slot = 2;
		while (scan < A.length) {
			if (A[scan] != A[slot - 1] || A[scan] != A[slot - 2]) {
				A[slot++] = A[scan++];
			} else {
				scan++;
			}
		}
		return slot;
	}

	// Minimum Path Sum
	public int minPathSum(int[][] grid) {
		if (grid == null || grid.length == 0 || grid[0].length == 0) {
			return -1;
		}
		int[][] memo = new int[grid.length][grid[0].length];
		memo[0][0] = grid[0][0];
		for (int j = 1; j < memo[0].length; j++) {
			memo[0][j] = grid[0][j] + memo[0][j - 1];
		}
		for (int i = 1; i < memo.length; i++) {
			memo[i][0] = grid[i][0] + memo[i - 1][0];
		}
		for (int i = 1; i < memo.length; i++) {
			for (int j = 1; j < memo[i].length; j++) {
				memo[i][j] = Math.min(memo[i - 1][j], memo[i][j - 1])
						+ grid[i][j];
			}
		}
		return memo[memo.length - 1][memo[0].length - 1];
	}

	// Rotate List
	public ListNode rotateRight(ListNode head, int n) {
		ListNode iter = head;
		int len = 0;
		while (iter != null) {
			len++;
			iter = iter.next;
		}
		n = len == 0 ? 0 : n % len;
		if (n == 0) {
			return head;
		}
		ListNode prev = null;
		ListNode start = head;
		ListNode end = head;
		for (int i = 0; i < n - 1; i++) {
			end = end.next;
		}
		while (end.next != null) {
			prev = start;
			start = start.next;
			end = end.next;
		}
		prev.next = null;
		end.next = head;
		return start;
	}

	// Text Justification
	public ArrayList<String> fullJustify(String[] words, int L) {
		if (words == null) {
			return null;
		}
		ArrayList<String> result = new ArrayList<String>();
		if (words.length == 0) {
			return result;
		}
		ArrayList<Integer> memo = new ArrayList<Integer>();
		memo.add(0);
		int index = 1;
		int len = words[0].length();
		while (index < words.length) {
			int wordLen = words[index].length();
			if (len + wordLen + 1 <= L) {
				len += (wordLen + 1);
			} else {
				len = wordLen;
				memo.add(index);
			}
			index++;
		}
		memo.add(index);
		for (int i = 0; i < memo.size() - 1; i++) {
			StringBuffer sb = new StringBuffer();
			int start = memo.get(i);
			int end = memo.get(i + 1) - 1;
			if (i == memo.size() - 2) {
				sb.append(words[start]);
				for (int j = start + 1; j <= end; j++) {
					sb.append(" " + words[j]);
				}
				while (sb.length() < L) {
					sb.append(" ");
				}
			} else {
				int sigma = 0;
				for (int j = start; j <= end; j++) {
					sigma += words[j].length();
				}
				int spaceNum = (end == start) ? (L - sigma) : (L - sigma)
						/ (end - start);
				StringBuffer space = new StringBuffer();
				for (int j = 0; j < spaceNum; j++) {
					space.append(" ");
				}
				int extra = (end == start) ? 0 : (L - sigma) % (end - start);
				for (int j = start; j <= end - 1; j++) {
					sb.append(words[j]);
					sb.append(space);
					if (extra > 0) {
						sb.append(" ");
						extra--;
					}
				}
				if (start == end) {
					sb.append(words[start]);
					sb.append(space);
				} else {
					sb.append(words[end]);
				}
			}
			result.add(sb.toString());
		}
		return result;
	}

	// CareerCup 6579701673885696
	public int[][] divide(int A[]) {
		if (A == null) {
			return null;
		}
		int sum = 0;
		for (int i : A) {
			sum += i;
		}
		HashSet<Integer> half = divideHelper(A, sum / 2, 0, A.length / 2);
		int[][] result = new int[A.length / 2][A.length / 2];
		int indexA = 0;
		int indexB = 0;
		for (int i = 0; i < A.length; i++) {
			if (half.contains(i)) {
				result[0][indexA++] = A[i];
			} else {
				result[1][indexB++] = A[i];
			}
		}
		return result;
	}

	private HashSet<Integer> divideHelper(int[] A, int target, int index, int n) {
		if (target == 0 && n == 0) {
			return new HashSet<Integer>();
		} else if (n == 0) {
			return null;
		}
		HashSet<Integer> result = null;
		for (int i = index; i < A.length; i++) {
			result = divideHelper(A, target - A[i], i + 1, n - 1);
			if (result != null) {
				result.add(i);
				break;
			}
		}
		return result;
	}

	// Median of Two Sorted Arrays of Equal Length
	public double findMedianSortedArraysEqualLength(int A[], int B[]) {
		if (A == null || B == null || A.length == 0 || B.length == 0
				|| A.length != B.length) {
			return Double.NaN;
		}
		return findMedianSortedArraysEqualLengthHelper(A, 0, A.length - 1, B,
				0, B.length - 1);
	}

	private double findMedianSortedArraysEqualLengthHelper(int A[], int leftA,
			int rightA, int[] B, int leftB, int rightB) {
		int lenA = rightA - leftA + 1;
		int lenB = rightB - leftB + 1;
		assert (lenA == lenB);
		int len = lenA;
		if (len == 1) {
			return (A[leftA] + B[leftB]) / 2D;
		} else if (len == 2) {
			int sum = Math.max(A[leftA], B[leftB])
					+ Math.min(A[rightA], B[rightB]);
			return sum / 2D;
		}
		int indexMidA = leftA + (rightA - leftA) / 2;
		int indexMidB = leftB + (rightB - leftB) / 2;
		int midA = (len % 2 == 0) ? (A[indexMidA] + A[indexMidA + 1]) / 2
				: A[indexMidA];
		int midB = (len % 2 == 0) ? (B[indexMidB] + B[indexMidB + 1]) / 2
				: B[indexMidB];
		if (midA == midB) {
			if (len % 2 == 0) {
				return findMedianSortedArraysEqualLengthHelper(A, indexMidA,
						indexMidA + 1, B, indexMidB, indexMidB + 1);
			} else {
				return findMedianSortedArraysEqualLengthHelper(A, indexMidA,
						indexMidA, B, indexMidB, indexMidB);
			}
		} else if (midA > midB) {
			if (len % 2 == 0) {
				return findMedianSortedArraysEqualLengthHelper(A, leftA,
						indexMidA + 1, B, indexMidB, rightB);
			} else {
				return findMedianSortedArraysEqualLengthHelper(A, leftA,
						indexMidA, B, indexMidB, rightB);
			}
		} else {
			if (len % 2 == 0) {
				return findMedianSortedArraysEqualLengthHelper(A, indexMidA,
						rightA, B, leftB, indexMidB + 1);
			} else {
				return findMedianSortedArraysEqualLengthHelper(A, indexMidA,
						rightA, B, leftB, indexMidB);
			}
		}
	}

	// Populating Next Right Pointers in Each Node
	public void connect(TreeLinkNode root) {
		if (root == null) {
			return;
		}
		TreeLinkNode prev = root;
		TreeLinkNode curr = root.left;
		while (curr != null) {
			TreeLinkNode iter1 = prev;
			TreeLinkNode iter2 = curr;
			while (iter1 != null) {
				iter2.next = iter1.right;
				iter1 = iter1.next;
				iter2 = iter2.next;
				if (iter1 != null) {
					iter2.next = iter1.left;
					iter2 = iter2.next;
				}
			}
			prev = curr;
			curr = curr.left;
		}
	}

	// Path Sum
	// Recursive solution
	public boolean hasPathSum(TreeNode root, int sum) {
		if (root == null) {
			return false;
		} else if (root.val == sum && root.left == null && root.right == null) {
			return true;
		} else {
			return hasPathSum(root.left, sum - root.val)
					|| hasPathSum(root.right, sum - root.val);
		}
	}

	// Path Sum
	// Iterative solution
	public boolean hasPathSumIter(TreeNode root, int sum) {
		Stack<TreeNode> stack = new Stack<TreeNode>();
		int accu = 0;
		while (root != null) {
			while (root != null) {
				stack.push(root);
				accu += root.val;
				root = root.left != null ? root.left : root.right;
			}
			if (accu == sum) {
				return true;
			}
			TreeNode child = null;
			root = stack.peek();
			while (!stack.isEmpty() && root != null
					&& (root.right == null || root.right == child)) {
				child = stack.pop();
				accu -= child.val;
				if (!stack.isEmpty()) {
					root = stack.peek();
				} else {
					root = null;
				}
			}
			root = root == null ? null : root.right;
		}
		return false;
	}

	// Binary Tree Inorder Traversal
	// Pop root immediately after been traversed
	public ArrayList<Integer> inorderTraversal(TreeNode root) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		Stack<TreeNode> stack = new Stack<TreeNode>();
		while (root != null) {
			while (root != null) {
				stack.push(root);
				root = root.left;
			}
			root = stack.pop();
			result.add(root.val);
			while (!stack.isEmpty() && root.right == null) {
				root = stack.pop();
				result.add(root.val);
			}
			root = root.right;
		}
		return result;
	}

	// Binary Tree Inorder Traversal
	// Same stack with preorder and postorder traversal
	public ArrayList<Integer> inorderTraversalTraditional(TreeNode root) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		Stack<TreeNode> stack = new Stack<TreeNode>();
		while (root != null) {
			while (root != null) {
				stack.push(root);
				if (root.left != null) {
					root = root.left;
				} else {
					result.add(root.val);
					root = root.right;
				}
			}
			TreeNode child = null;
			root = stack.peek();
			while (!stack.isEmpty() && root != null
					&& (root.right == null || root.right == child)) {
				if (root.right != child) {
					result.add(root.val);
				}
				child = stack.pop();
				if (!stack.isEmpty()) {
					root = stack.peek();
				} else {
					root = null;
				}
			}
			if (root != null) {
				result.add(root.val);
				root = root.right;
			}
		}
		return result;
	}

	// Combination Sum
	public ArrayList<ArrayList<Integer>> combinationSum(int[] candidates,
			int target) {
		Arrays.sort(candidates);
		return combinationSumHelper(candidates, target, candidates.length - 1);
	}

	private ArrayList<ArrayList<Integer>> combinationSumHelper(
			int[] candidates, int target, int index) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (target == 0) {
			ArrayList<Integer> empty = new ArrayList<Integer>();
			result.add(empty);
			return result;
		} else if (index < 0 || target < 0) {
			return null;
		}
		for (int i = 0; i <= target / candidates[index]; i++) {
			ArrayList<ArrayList<Integer>> partialResult = combinationSumHelper(
					candidates, target - i * candidates[index], index - 1);
			if (partialResult != null) {
				for (ArrayList<Integer> al : partialResult) {
					for (int j = 0; j < i; j++) {
						al.add(candidates[index]);
					}
				}
				result.addAll(partialResult);
			}
		}
		return result;
	}

	// 4Sum
	public ArrayList<ArrayList<Integer>> fourSum(int[] num, int target) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		Arrays.sort(num);
		for (int i = 0; i < num.length - 3; i++) {
			if (i > 0 && num[i] == num[i - 1]) {
				continue;
			}
			for (int j = i + 1; j < num.length - 2; j++) {
				if (j > i + 1 && num[j] == num[j - 1]) {
					continue;
				}
				int left = j + 1;
				int right = num.length - 1;
				int rest = target - num[i] - num[j];
				while (left < right) {
					int sum = num[left] + num[right];
					if (sum == rest) {
						ArrayList<Integer> found = new ArrayList<Integer>();
						found.add(num[i]);
						found.add(num[j]);
						found.add(num[left]);
						found.add(num[right]);
						result.add(found);
						do {
							left++;
						} while (left < num.length
								&& num[left] == num[left - 1]);
						do {
							right--;
						} while (right >= 0 && num[right] == num[right + 1]);
					} else if (sum < rest) {
						left++;
					} else {
						right--;
					}
				}
			}
		}
		return result;
	}

	// Symmetric Tree
	public boolean isSymmetric(TreeNode root) {
		if (root == null) {
			return true;
		}
		return isSymmetricHelper(root.left, root.right);
	}

	private boolean isSymmetricHelper(TreeNode root1, TreeNode root2) {
		if (root1 == null && root2 == null) {
			return true;
		} else if (root1 == null || root2 == null) {
			return false;
		} else if (root1.val == root2.val) {
			return isSymmetricHelper(root1.left, root2.right)
					&& isSymmetricHelper(root1.right, root2.left);
		} else {
			return false;
		}
	}

	// Reverse Nodes in k-Group
	public ListNode reverseKGroup(ListNode head, int k) {
		if (head == null || k < 2) {
			return head;
		}
		ListNode finalHead = head;
		ListNode prev = null;
		ListNode start = head;
		ListNode end = head;
		ListNode post = null;
		while (end != null) {
			for (int i = 0; i < k - 1 && end != null; i++) {
				end = end.next;
			}
			if (end != null) {
				if (finalHead == head) {
					finalHead = end;
				}
				post = end.next;
				reverseKGroupHelper(start, end);
				if (prev != null) {
					prev.next = end;
				}
				start.next = post;
				prev = start;
				start = post;
				end = post;
			}
		}
		return finalHead;
	}

	private void reverseKGroupHelper(ListNode start, ListNode end) {
		if (start == end) {
			return;
		}
		ListNode prev = start;
		ListNode curr = prev.next;
		ListNode post = curr.next;
		while (prev != end) {
			curr.next = prev;
			prev = curr;
			curr = post;
			post = post == null ? null : post.next;
		}
	}

	// Candy
	public int candy(int[] ratings) {
		int candies[] = new int[ratings.length];
		Arrays.fill(candies, 0);
		candies[0] = 1;
		for (int i = 0; i < candies.length - 1; i++) {
			if (ratings[i] < ratings[i + 1] && candies[i] >= candies[i + 1]) {
				candies[i + 1] = candies[i] + 1;
			} else {
				candies[i + 1] = 1;
			}
		}
		for (int i = candies.length - 1; i > 0; i--) {
			if (ratings[i] < ratings[i - 1] && candies[i] >= candies[i - 1]) {
				candies[i - 1] = candies[i] + 1;
			}
		}
		int sum = 0;
		for (int i = 0; i < candies.length; i++) {
			sum += candies[i];
		}
		return sum;
	}

	// Maximum Subarray
	public int maxSubArray(int[] A) {
		if (A == null || A.length == 0) {
			return 0;
		}
		int sum = 0;
		int max = A[0];
		for (int i = 0; i < A.length; i++) {
			sum += A[i];
			max = Math.max(max, sum);
			sum = sum < 0 ? 0 : sum;
		}
		return max;
	}

	// Pascal's Triangle
	public ArrayList<ArrayList<Integer>> generate(int numRows) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (numRows <= 0) {
			return result;
		}
		ArrayList<Integer> prev = new ArrayList<Integer>();
		prev.add(1);
		result.add(prev);
		for (int i = 1; i < numRows; i++) {
			ArrayList<Integer> curr = new ArrayList<Integer>();
			for (int j = 0; j <= prev.size(); j++) {
				int before = j == 0 ? 0 : prev.get(j - 1);
				int now = j == prev.size() ? 0 : prev.get(j);
				curr.add(before + now);
			}
			result.add(curr);
			prev = curr;
		}
		return result;
	}

	// Unique Binary Search Trees
	public int numTrees(int n) {
		if (n < 0) {
			return 0;
		}
		int[] memo = new int[n + 1];
		Arrays.fill(memo, 0);
		memo[0] = 1;
		memo[1] = 1;
		for (int i = 2; i <= n; i++) {
			for (int j = 0; j < i; j++) {
				memo[i] += memo[j] * memo[i - j - 1];
			}
		}
		return memo[n];
	}

	// Binary Tree Level Order Traversal II
	public ArrayList<ArrayList<Integer>> levelOrderBottom(TreeNode root) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (root == null) {
			return result;
		}
		Stack<ArrayList<Integer>> stack = new Stack<ArrayList<Integer>>();
		Queue<TreeNode> queue = new ArrayDeque<TreeNode>();
		queue.offer(root);
		int levelNum = 1;
		while (!queue.isEmpty()) {
			ArrayList<Integer> level = new ArrayList<Integer>();
			for (int i = 0; i < levelNum; i++) {
				TreeNode node = queue.poll();
				if (node.left != null) {
					queue.offer(node.left);
				}
				if (node.right != null) {
					queue.offer(node.right);
				}
				level.add(node.val);
			}
			stack.push(level);
			levelNum = queue.size();
		}
		while (!stack.isEmpty()) {
			result.add(stack.pop());
		}
		return result;
	}

	// Integer to Roman
	public String intToRoman(int num) {
		String[][] table = new String[][] {
				{ "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX" },
				{ "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC" },
				{ "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM" },
				{ "M", "MM", "MMM", null, null, null, null, null, null } };
		String digitString = null;
		String result = new String();
		int row = 0;
		int digit = 0;
		while (num > 0) {
			digit = num % 10;
			if (digit != 0) {
				digitString = table[row][digit - 1];
				result = digitString + result;
			}
			num /= 10;
			row++;
		}
		return result;
	}

	// Insertion Sort List
	public ListNode insertionSortList(ListNode head) {
		ListNode start = head;
		ListNode end = head;
		while (end != null && end.next != null) {
			ListNode node = end.next;
			if (node.val < start.val) {
				end.next = node.next;
				node.next = start;
				start = node;
			} else {
				ListNode scan = start;
				while (scan != end && scan.next.val <= node.val) {
					scan = scan.next;
				}
				if (scan != end) {
					end.next = node.next;
					node.next = scan.next;
					scan.next = node;
				} else {
					end = end.next;
				}
			}
		}
		return start;
	}

	// Insert Interval
	public ArrayList<Interval> insert(ArrayList<Interval> intervals,
			Interval newInterval) {
		int index1 = 0;
		while (index1 < intervals.size()
				&& intervals.get(index1).end < newInterval.start) {
			index1++;
		}
		if (index1 == intervals.size()) {
			intervals.add(newInterval);
		} else {
			int index2 = index1;
			while (index2 < intervals.size()
					&& intervals.get(index2).start <= newInterval.end) {
				index2++;
			}
			index2--;
			if (index2 < index1) {
				intervals.add(index1, newInterval);
			} else {
				Interval update = intervals.get(index2);
				int newStart = Math.min(intervals.get(index1).start,
						newInterval.start);
				int newEnd = Math.max(update.end, newInterval.end);
				intervals.subList(index1, index2).clear();
				update.start = newStart;
				update.end = newEnd;
			}
		}
		return intervals;
	}

	// LRU Cache
	private class CacheNode {
		private int key;
		private int val;
		private CacheNode prev;
		private CacheNode next;

		public CacheNode(int key, int val, CacheNode prev, CacheNode next) {
			this.key = key;
			this.val = val;
			this.prev = prev;
			this.next = next;
		}
	}

	public class LRUCache {
		private int capacity;
		private int size;
		private CacheNode head;
		private CacheNode tail;
		private HashMap<Integer, CacheNode> memo;

		public LRUCache(int capacity) {
			this.capacity = capacity;
			this.size = 0;
			this.head = new CacheNode(-1, -1, null, null);
			this.tail = new CacheNode(-1, -1, this.head, null);
			this.head.next = tail;
			this.memo = new HashMap<Integer, CacheNode>();
		}

		public int get(int key) {
			int val = -1;
			if (memo.containsKey(key)) {
				CacheNode cn = memo.get(key);
				val = cn.val;
				refresh(cn);
			}
			return val;
		}

		public void set(int key, int value) {
			if (memo.containsKey(key)) {
				CacheNode cn = memo.get(key);
				cn.val = value;
				refresh(cn);
			} else {
				if (size < capacity) {
					CacheNode cn = new CacheNode(key, value, head, head.next);
					head.next.prev = cn;
					head.next = cn;
					memo.put(key, cn);
					size++;
				} else {
					memo.remove(tail.prev.key);
					tail.prev.key = key;
					memo.put(key, tail.prev);
					tail.prev.val = value;
					refresh(tail.prev);
				}
			}
		}

		private void refresh(CacheNode cn) {
			cn.prev.next = cn.next;
			cn.next.prev = cn.prev;
			cn.next = head.next;
			cn.prev = head;
			head.next.prev = cn;
			head.next = cn;
		}
	}

	// Swap Nodes in Pairs
	public ListNode swapPairs(ListNode head) {
		if (head == null || head.next == null) {
			return head;
		}
		ListNode finalHead = head.next;
		ListNode prev = null;
		ListNode node1 = head;
		ListNode node2 = head.next;
		ListNode post = node2.next;
		while (node2 != null) {
			node2.next = node1;
			if (prev != null) {
				prev.next = node2;
			}
			node1.next = post;
			prev = node1;
			node1 = post;
			node2 = node1 == null ? null : node1.next;
			post = node2 == null ? null : node2.next;
		}
		return finalHead;
	}

	// 3Sum
	public ArrayList<ArrayList<Integer>> threeSum(int[] num) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		Arrays.sort(num);
		for (int i = 0; i < num.length - 2; i++) {
			if (i > 0 && num[i] == num[i - 1]) {
				continue;
			}
			int left = i + 1;
			int right = num.length - 1;
			int target = -num[i];
			while (left < right) {
				int sum = num[left] + num[right];
				if (sum == target) {
					ArrayList<Integer> found = new ArrayList<Integer>();
					found.add(num[i]);
					found.add(num[left]);
					found.add(num[right]);
					result.add(found);
					do {
						left++;
					} while (left < num.length && num[left] == num[left - 1]);
					do {
						right--;
					} while (right >= 0 && num[right] == num[right + 1]);
				} else if (sum > target) {
					right--;
				} else {
					left++;
				}
			}
		}
		return result;
	}

	// Copy List with Random Pointer
	public RandomListNode copyRandomList(RandomListNode head) {
		HashMap<RandomListNode, RandomListNode> memo = new HashMap<RandomListNode, RandomListNode>();
		RandomListNode copyHead = null;
		RandomListNode copyTail = null;
		RandomListNode iter = head;
		while (iter != null) {
			if (copyTail == null) {
				copyTail = new RandomListNode(iter.label);
				copyHead = copyTail;
			} else {
				copyTail.next = new RandomListNode(iter.label);
				copyTail = copyTail.next;
			}
			memo.put(iter, copyTail);
			iter = iter.next;
		}
		iter = head;
		RandomListNode copyIter = copyHead;
		while (iter != null) {
			if (iter.random == null) {
				copyIter.random = null;
			} else {
				copyIter.random = memo.get(iter.random);
			}
			iter = iter.next;
			copyIter = copyIter.next;
		}
		return copyHead;
	}

	public RandomListNode copyRandomListInPlace(RandomListNode head) {
		if (head == null) {
			return null;
		}
		RandomListNode curr = head;
		RandomListNode post = head.next;
		while (curr != null) {
			curr.next = new RandomListNode(curr.label);
			curr.next.next = post;
			curr = post;
			post = (post == null) ? null : post.next;
		}
		RandomListNode iter = head;
		RandomListNode copyIter = head.next;
		while (iter != null) {
			if (iter.random == null) {
				copyIter.random = null;
			} else {
				copyIter.random = iter.random.next;
			}
			iter = iter.next.next;
			copyIter = (copyIter.next == null) ? null : copyIter.next.next;
		}
		RandomListNode tail = head;
		RandomListNode copyHead = head.next;
		RandomListNode copyTail = head.next;
		while (tail != null) {
			tail.next = copyTail.next;
			copyTail.next = (copyTail.next == null) ? null : copyTail.next.next;
			tail = tail.next;
			copyTail = copyTail.next;
		}
		return copyHead;
	}

	// Remove Nth Node From End of List
	public ListNode removeNthFromEnd(ListNode head, int n) {
		if (head == null) {
			return null;
		}
		ListNode iter1 = head;
		ListNode iter2 = head;
		for (int i = 0; i < n; i++) {
			iter2 = iter2.next;
		}
		if (iter2 == null) {
			ListNode ret = head.next;
			head.next = null;
			return ret;
		}
		while (iter2.next != null) {
			iter1 = iter1.next;
			iter2 = iter2.next;
		}
		iter1.next = iter1.next.next;
		return head;
	}

	// Word Break
	public boolean wordBreak(String s, Set<String> dict) {
		boolean[] memo = new boolean[s.length() + 1];
		Arrays.fill(memo, false);
		memo[s.length()] = true;
		int end = s.length() - 1;
		for (int start = end; start >= 0; start--) {
			for (int i = 1; i <= end - start + 1; i++) {
				if (dict.contains(s.substring(start, start + i))
						&& memo[start + i]) {
					memo[start] = true;
				}
			}
		}
		return memo[0];
	}

	// Gray Code
	public ArrayList<Integer> grayCode(int n) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		if (n < 0) {
			return result;
		}
		result.add(0);
		if (n == 0) {
			return result;
		}
		result.add(1);
		if (n == 1) {
			return result;
		}
		for (int i = 1; i < n; i++) {
			int len = result.size();
			for (int j = len - 1; j >= 0; j--) {
				result.add(result.get(j) + (int) Math.pow(2, i));
			}
		}
		return result;
	}

	// Search in Rotated Sorted Array
	public int search(int[] A, int target) {
		int left = 0;
		int right = A.length - 1;
		while (left <= right) {
			int mid = left + (right - left) / 2;
			if (A[mid] == target) {
				return mid;
			} else if (A[left] <= A[mid]) {
				if (A[left] <= target && A[mid] > target) {
					right = mid - 1;
				} else {
					left = mid + 1;
				}
			} else if (A[mid] <= A[right]) {
				if (A[mid] < target && A[right] >= target) {
					left = mid + 1;
				} else {
					right = mid - 1;
				}
			}
		}
		return -1;
	}

	// Search in Rotated Sorted Array II
	public boolean searchII(int[] A, int target) {
		return searchHelper(A, target, 0, A.length - 1);
	}

	private boolean searchHelper(int[] A, int target, int left, int right) {
		if (left > right) {
			return false;
		}
		int mid = left + (right - left) / 2;
		if (A[mid] == target) {
			return true;
		} else if (A[left] == A[mid] && A[mid] == A[right]) {
			return searchHelper(A, target, left, mid - 1)
					|| searchHelper(A, target, mid + 1, right);
		} else if (A[left] <= A[mid]) {
			if (A[left] <= target && A[mid] > target) {
				return searchHelper(A, target, left, mid - 1);
			} else {
				return searchHelper(A, target, mid + 1, right);
			}
		} else {
			if (A[mid] < target && A[right] >= target) {
				return searchHelper(A, target, mid + 1, right);
			} else {
				return searchHelper(A, target, left, mid - 1);
			}
		}
	}

	// Binary Tree Level Order Traversal
	public ArrayList<ArrayList<Integer>> levelOrder(TreeNode root) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (root == null) {
			return result;
		}
		Queue<TreeNode> queue = new ArrayDeque<TreeNode>();
		queue.offer(root);
		int levelNum = 1;
		while (!queue.isEmpty()) {
			ArrayList<Integer> level = new ArrayList<Integer>();
			for (int i = 0; i < levelNum; i++) {
				TreeNode node = queue.poll();
				if (node.left != null) {
					queue.offer(node.left);
				}
				if (node.right != null) {
					queue.offer(node.right);
				}
				level.add(node.val);
			}
			result.add(level);
			levelNum = queue.size();
		}
		return result;
	}

	// Partition List
	public ListNode partition(ListNode head, int x) {
		ListNode smallHead = head;
		ListNode smallTail = null;
		ListNode bigHead = null;
		ListNode bigTail = null;
		while (head != null) {
			if (head.val < x) {
				if (smallTail == null) {
					smallTail = head;
					smallHead = head;
				} else {
					smallTail.next = head;
					smallTail = smallTail.next;
				}
			} else {
				if (bigTail == null) {
					bigTail = head;
					bigHead = head;
				} else {
					bigTail.next = head;
					bigTail = bigTail.next;
				}
			}
			head = head.next;
		}
		if (smallTail != null) {
			smallTail.next = bigHead;
		}
		if (bigTail != null) {
			bigTail.next = null;
		}
		return smallHead;
	}

	// Flatten Binary Tree to Linked List
	public void flatten(TreeNode root) {
		flattenHelper(root);
	}

	TreeNode[] flattenHelper(TreeNode root) {
		if (root == null) {
			return null;
		}
		TreeNode[] leftSide = flattenHelper(root.left);
		TreeNode[] rightSide = flattenHelper(root.right);
		TreeNode end = root;
		end.left = null;
		end.right = null;
		if (leftSide != null) {
			end.right = leftSide[0];
			end = leftSide[1];
		}
		if (rightSide != null) {
			end.right = rightSide[0];
			end = rightSide[1];
		}
		return new TreeNode[] { root, end };
	}

	// Edit Distance
	public int minDistance(String word1, String word2) {
		if (word1 == null || word2 == null) {
			return -1;
		}
		int[][] memo = new int[word1.length() + 1][word2.length() + 1];
		for (int i = 0; i < memo.length; i++) {
			memo[i][0] = i;
		}
		for (int j = 0; j < memo[0].length; j++) {
			memo[0][j] = j;
		}
		for (int i = 1; i < memo.length; i++) {
			for (int j = 1; j < memo[i].length; j++) {
				if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
					memo[i][j] = memo[i - 1][j - 1];
				} else {
					int min = Math.min(
							Math.min(memo[i][j - 1], memo[i - 1][j]),
							memo[i - 1][j - 1]);
					memo[i][j] = min + 1;
				}
			}
		}
		return memo[memo.length - 1][memo[0].length - 1];
	}

	// Implement strStr()
	public String strStr(String haystack, String needle) {
		if (haystack == null || needle == null) {
			return null;
		}
		if (needle.isEmpty()) {
			return haystack;
		}
		int start = 0;
		int end = haystack.length() - needle.length();
		while (start <= end) {
			int nextStart = -1;
			if (haystack.charAt(start) == needle.charAt(0)) {
				int index = 1;
				while (index < needle.length()
						&& haystack.charAt(start + index) == needle
								.charAt(index)) {
					if (nextStart == -1
							&& haystack.charAt(start + index) == needle
									.charAt(0)) {
						nextStart = start + index;
					}
					index++;
				}
				if (index == needle.length()) {
					return haystack.substring(start, haystack.length());
				}
			}
			if (nextStart != -1) {
				start = nextStart;
			} else {
				start++;
			}
		}
		return null;
	}

	// Distinct Subsequences
	public int numDistinct(String S, String T) {
		if (S == null || T == null || S.length() < T.length()) {
			return 0;
		}
		int[][] memo = new int[S.length() + 1][T.length() + 1];
		for (int i = 0; i < memo.length; i++) {
			memo[i][0] = 1;
		}
		for (int j = 1; j < memo[0].length; j++) {
			memo[0][j] = 0;
		}
		for (int i = 1; i < memo.length; i++) {
			for (int j = 1; j <= Math.min(i, T.length()); j++) {
				if (S.charAt(i - 1) == T.charAt(j - 1)) {
					memo[i][j] = memo[i - 1][j] + memo[i - 1][j - 1];
				} else {
					memo[i][j] = memo[i - 1][j];
				}
			}
		}
		return memo[memo.length - 1][memo[0].length - 1];
	}

	// Roman to Integer
	public int romanToInt(String s) {
		if (s == null || s.isEmpty()) {
			return 0;
		}
		HashMap<Character, Integer> table = new HashMap<Character, Integer>();
		table.put('I', 1);
		table.put('V', 5);
		table.put('X', 10);
		table.put('L', 50);
		table.put('C', 100);
		table.put('D', 500);
		table.put('M', 1000);
		int prev = Integer.MIN_VALUE;
		int result = 0;
		for (int i = s.length() - 1; i >= 0; i--) {
			char c = s.charAt(i);
			if (!table.containsKey(c)) {
				return 0;
			}
			int value = table.get(c);
			if (value < prev) {
				result -= value;
			} else {
				result += value;
			}
			prev = value;
		}
		return result;
	}

	// Clone Graph
	public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
		if (node == null) {
			return null;
		}
		HashMap<UndirectedGraphNode, UndirectedGraphNode> cloned = new HashMap<UndirectedGraphNode, UndirectedGraphNode>();
		return cloneGraphHelper(node, cloned);
	}

	private UndirectedGraphNode cloneGraphHelper(UndirectedGraphNode node,
			HashMap<UndirectedGraphNode, UndirectedGraphNode> cloned) {
		if (cloned.containsKey(node)) {
			return cloned.get(node);
		}
		UndirectedGraphNode newNode = new UndirectedGraphNode(node.label);
		cloned.put(node, newNode);
		for (UndirectedGraphNode ugn : node.neighbors) {
			newNode.neighbors.add(cloneGraphHelper(ugn, cloned));
		}
		return newNode;
	}

	// Wildcard Matching
	public boolean isMatchWildcard(String s, String p) {
		if (s == null || p == null) {
			return false;
		}
		int notStar = 0;
		for (int i = 0; i < p.length(); i++) {
			if (p.charAt(i) != '*') {
				notStar++;
			}
		}
		if (notStar > s.length()) {
			return false;
		}
		boolean[][] memo = new boolean[s.length() + 1][p.length() + 1];
		memo[s.length()][p.length()] = true;
		for (int i = s.length() - 1; i >= 0; i--) {
			memo[i][p.length()] = false;
		}
		boolean allStar = true;
		for (int j = p.length() - 1; j >= 0; j--) {
			if (allStar && p.charAt(j) != '*') {
				allStar = false;
			}
			memo[s.length()][j] = allStar;
		}
		for (int i = s.length() - 1; i >= 0; i--) {
			for (int j = p.length() - 1; j >= 0; j--) {
				if (p.charAt(j) == '?') {
					memo[i][j] = memo[i + 1][j + 1];
				} else if (p.charAt(j) == '*') {
					memo[i][j] = false;
					for (int k = i; k < memo.length; k++) {
						if (memo[k][j + 1]) {
							memo[i][j] = true;
							break;
						}
					}
				} else if (p.charAt(j) == s.charAt(i)) {
					memo[i][j] = memo[i + 1][j + 1];
				} else {
					memo[i][j] = false;
				}
			}
		}
		return memo[0][0];
	}

	// N-Queens II
	public int totalNQueens(int n) {
		int[] memo = new int[n];
		Arrays.fill(memo, 0);
		return totalNQueensHelper(memo, 0);
	}

	private int totalNQueensHelper(int[] memo, int row) {
		if (row == memo.length) {
			return 1;
		}
		int result = 0;
		for (int i = 0; i < memo.length; i++) {
			memo[row] = i;
			if (valid(memo, row)) {
				result += totalNQueensHelper(memo, row + 1);
			}
		}
		return result;
	}

	// Validate Binary Search Tree
	public boolean isValidBST(TreeNode root) {
		return isValidBSTHelper(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
	}

	private boolean isValidBSTHelper(TreeNode root, int min, int max) {
		if (root == null) {
			return true;
		} else if (root.val <= min || root.val >= max) {
			return false;
		} else {
			return isValidBSTHelper(root.left, min, root.val)
					&& isValidBSTHelper(root.right, root.val, max);
		}
	}

	// Climbing Stairs
	public int climbStairs(int n) {
		if (n < 0) {
			return 0;
		}
		if (n <= 1) {
			return 1;
		}
		int prevPrev = 1;
		int prev = 1;
		int curr = 0;
		for (int i = 2; i <= n; i++) {
			curr = prev + prevPrev;
			prevPrev = prev;
			prev = curr;
		}
		return curr;
	}

	// Subsets
	public ArrayList<ArrayList<Integer>> subsets(int[] S) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (S == null) {
			return result;
		}
		ArrayList<Integer> empty = new ArrayList<Integer>();
		result.add(empty);
		if (S.length == 0) {
			return result;
		}
		Arrays.sort(S);
		for (int i = 0; i < S.length; i++) {
			int size = result.size();
			for (int j = 0; j < size; j++) {
				ArrayList<Integer> al = new ArrayList<Integer>(result.get(j));
				al.add(S[i]);
				result.add(al);
			}
		}
		return result;
	}

	// Maximal Rectangle
	public int maximalRectangle(char[][] matrix) {
		if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
			return 0;
		}
		int[][] memo = new int[matrix.length][matrix[0].length];
		for (int i = 0; i < memo.length; i++) {
			for (int j = 0; j < memo[i].length; j++) {
				memo[i][j] = (i == 0) ? (matrix[i][j] - '0')
						: (matrix[i][j] - '0' + memo[i - 1][j]);
			}
		}
		int max = 0;
		for (int row1 = 0; row1 < matrix.length; row1++) {
			for (int row2 = row1; row2 < matrix.length; row2++) {
				int len = row2 - row1 + 1;
				int col = 0;
				while (col < matrix[0].length) {
					int colVal = (row1 == 0) ? (memo[row2][col])
							: (memo[row2][col] - memo[row1 - 1][col]);
					int nextCol = col + 1;
					if (colVal == len) {
						while (nextCol < matrix[0].length) {
							int nextColVal = (row1 == 0) ? (memo[row2][nextCol])
									: (memo[row2][nextCol] - memo[row1 - 1][nextCol]);
							if (nextColVal == len) {
								nextCol++;
							} else {
								break;
							}
						}
						max = Math.max(max, (nextCol - col) * len);
					}
					col = nextCol;
				}
			}
		}
		return max;
	}

	// Generate Parentheses
	public ArrayList<String> generateParenthesis(int n) {
		StringBuffer sb = new StringBuffer();
		ArrayList<String> result = new ArrayList<String>();
		generateParenthesisHelper(n, n, sb, result);
		return result;
	}

	private void generateParenthesisHelper(int left, int right,
			StringBuffer sb, ArrayList<String> result) {
		if (left == 0 && right == 0) {
			result.add(sb.toString());
			return;
		} else {
			if (left < right) {
				generateParenthesisHelper(left, right - 1, sb.append(")"),
						result);
				sb.deleteCharAt(sb.length() - 1);
			}
			if (left > 0) {
				generateParenthesisHelper(left - 1, right, sb.append("("),
						result);
				sb.deleteCharAt(sb.length() - 1);
			}
		}
	}

	// Valid Palindrome
	public boolean isPalindrome(String s) {
		if (s == null) {
			return false;
		}
		int left = 0;
		int right = s.length() - 1;
		while (left < right) {
			while (left < right
					&& !Character.isLetterOrDigit(Character.toLowerCase(s
							.charAt(left)))) {
				left++;
			}
			while (left < right
					&& !Character.isLetterOrDigit(Character.toLowerCase(s
							.charAt(right)))) {
				right--;
			}
			if (left < right) {
				if (Character.toLowerCase(s.charAt(left)) == Character
						.toLowerCase(s.charAt(right))) {
					left++;
					right--;
				} else {
					return false;
				}
			}
		}
		return true;
	}

	// Search a 2D Matrix
	public boolean searchMatrix(int[][] matrix, int target) {
		if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
			return false;
		}
		int left = 0;
		int right = matrix.length * matrix[0].length - 1;
		while (left <= right) {
			int mid = left + (right - left) / 2;
			int row = mid / matrix[0].length;
			int col = mid % matrix[0].length;
			if (matrix[row][col] == target) {
				return true;
			} else if (matrix[row][col] > target) {
				right = mid - 1;
			} else {
				left = mid + 1;
			}
		}
		return false;
	}

	// Plus One
	public int[] plusOne(int[] digits) {
		if (digits == null || digits.length == 0) {
			int[] result = new int[1];
			result[0] = 1;
			return result;
		}
		int index = digits.length - 1;
		while (index >= 0 && digits[index] == 9) {
			index--;
		}
		if (index < 0) {
			int[] result = new int[digits.length + 1];
			Arrays.fill(result, 0);
			result[0] = 1;
			return result;
		} else {
			digits[index] += 1;
			for (int i = index + 1; i < digits.length; i++) {
				digits[i] = 0;
			}
		}
		return digits;
	}

	// Longest Palindromic Substring
	public String longestPalindrome(String s) {
		if (s == null || s.isEmpty()) {
			return null;
		}
		int start = 0;
		int end = 0;
		boolean[][] memo = new boolean[s.length()][s.length()];
		for (int i = 0; i < memo.length; i++) {
			memo[i][i] = true;
		}
		for (int i = 0; i < memo.length - 1; i++) {
			if (s.charAt(i) == s.charAt(i + 1)) {
				memo[i][i + 1] = true;
				start = i;
				end = i + 1;
			} else {
				memo[i][i + 1] = false;
			}
		}
		for (int len = 3; len <= s.length(); len++) {
			for (int row = 0; row <= s.length() - len; row++) {
				int col = row + len - 1;
				if (s.charAt(row) == s.charAt(col)) {
					memo[row][col] = memo[row + 1][col - 1];
					if (memo[row][col] && (len > end - start + 1)) {
						start = row;
						end = col;
					}
				} else {
					memo[row][col] = false;
				}
			}
		}
		return s.substring(start, end + 1);
	}

	// Container With Most Water
	public int maxArea(int[] height) {
		int left = 0;
		int right = height.length - 1;
		int max = 0;
		while (left < right) {
			int vol = Math.min(height[left], height[right]) * (right - left);
			max = Math.max(max, vol);
			if (height[left] < height[right]) {
				left++;
			} else {
				right--;
			}
		}
		return max;
	}

	// Spiral Matrix
	public ArrayList<Integer> spiralOrder(int[][] matrix) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
			return result;
		}
		int m = matrix.length;
		int n = matrix[0].length;
		for (int i = 0; i <= Math.min((n - 1) / 2, (m - 1) / 2); i++) {
			int row1 = i;
			int col1 = i;
			int row2 = m - i - 1;
			int col2 = n - i - 1;
			if (row1 == row2) {
				for (int col = col1; col <= col2; col++) {
					result.add(matrix[row1][col]);
				}
			} else if (col1 == col2) {
				for (int row = row1; row <= row2; row++) {
					result.add(matrix[row][col1]);
				}
			} else {
				for (int col = col1; col <= col2; col++) {
					result.add(matrix[row1][col]);
				}
				for (int row = row1 + 1; row <= row2; row++) {
					result.add(matrix[row][col2]);
				}
				for (int col = col2 - 1; col >= col1; col--) {
					result.add(matrix[row2][col]);
				}
				for (int row = row2 - 1; row > row1; row--) {
					result.add(matrix[row][col1]);
				}
			}
		}
		return result;
	}

	// Maximum Depth of Binary Tree
	// Recursive solution
	public int maxDepth(TreeNode root) {
		if (root == null) {
			return 0;
		}
		return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
	}

	// Maximum Depth of Binary Tree
	// Iterative solution
	public int maxDepthIter(TreeNode root) {
		Stack<TreeNode> stack = new Stack<TreeNode>();
		int depth = 0;
		int max = 0;
		while (root != null) {
			while (root != null) {
				stack.push(root);
				depth++;
				root = root.left != null ? root.left : root.right;
			}
			max = Math.max(max, depth);
			TreeNode child = null;
			root = stack.peek();
			while (!stack.isEmpty() && root != null
					&& (root.right == null || root.right == child)) {
				child = stack.pop();
				depth--;
				if (!stack.isEmpty()) {
					root = stack.peek();
				} else {
					root = null;
				}
			}
			root = root == null ? null : root.right;
		}
		return max;
	}

	// Search Insert Position
	public int searchInsert(int[] A, int target) {
		int left = 0;
		int right = A.length - 1;
		int mid = 0;
		while (left <= right) {
			mid = left + (right - left) / 2;
			if (A[mid] == target) {
				return mid;
			} else if (A[mid] < target) {
				left = mid + 1;
			} else {
				right = mid - 1;
			}
		}
		if (A[mid] < target) {
			return mid + 1;
		} else {
			return mid;
		}
	}

	// Best Time to Buy and Sell Stock
	public int maxProfit(int[] prices) {
		if (prices == null || prices.length < 2) {
			return 0;
		}
		int maxPrice = prices[prices.length - 1];
		int maxProfit = 0;
		for (int i = prices.length - 2; i >= 0; i--) {
			maxProfit = Math.max(maxProfit, maxPrice - prices[i]);
			maxPrice = Math.max(maxPrice, prices[i]);
		}
		return maxProfit;
	}

	// Merge Intervals
	public ArrayList<Interval> merge(ArrayList<Interval> intervals) {
		if (intervals == null || intervals.size() < 2) {
			return intervals;
		}
		Collections.sort(intervals, new Comparator<Interval>() {
			public int compare(Interval i1, Interval i2) {
				return i1.start - i2.start;
			}
		});
		int index = 0;
		while (index < intervals.size()) {
			int toMerge = index + 1;
			int maxEnd = intervals.get(index).end;
			while (toMerge < intervals.size()
					&& intervals.get(index).end >= intervals.get(toMerge).start) {
				maxEnd = Math.max(maxEnd, intervals.get(toMerge).end);
				toMerge++;
			}
			if (toMerge > index + 1) {
				intervals.get(index).end = maxEnd;
				intervals.subList(index + 1, toMerge).clear();
			} else {
				index++;
			}
		}
		return intervals;
	}

	// Merge Intervals
	// Another solution is to merge one by one
	public ArrayList<Interval> mergeOneByOne(ArrayList<Interval> intervals) {
		Collections.sort(intervals, new Comparator<Interval>() {
			public int compare(Interval i1, Interval i2) {
				return i1.start - i2.start;
			}
		});
		int index = 0;
		while (index < intervals.size() - 1) {
			Interval i1 = intervals.get(index);
			Interval i2 = intervals.get(index + 1);
			if (i1.end >= i2.start) {
				i1.end = Math.max(i1.end, i2.end);
				intervals.remove(index + 1);
			} else {
				index++;
			}
		}
		return intervals;
	}

	// Balanced Binary Tree
	public boolean isBalanced(TreeNode root) {
		return isBalancedHelper(root) != -1;
	}

	private int isBalancedHelper(TreeNode root) {
		if (root == null) {
			return 0;
		}
		int left = isBalancedHelper(root.left);
		int right = isBalancedHelper(root.right);
		if (left == -1 || right == -1 || Math.abs(left - right) > 1) {
			return -1;
		} else {
			return Math.max(left, right) + 1;
		}
	}

	// Path Sum II
	// Recursive solution
	public ArrayList<ArrayList<Integer>> pathSum(TreeNode root, int sum) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> path = new ArrayList<Integer>();
		pathSumHelper(root, sum, path, result);
		return result;
	}

	// Path Sum II
	// Iterative solution
	public ArrayList<ArrayList<Integer>> pathSumIter(TreeNode root, int sum) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		Stack<TreeNode> stack = new Stack<TreeNode>();
		int accu = 0;
		while (root != null) {
			while (root != null) {
				stack.push(root);
				accu += root.val;
				root = root.left != null ? root.left : root.right;
			}
			if (accu == sum) {
				ArrayList<TreeNode> path = new ArrayList<TreeNode>(stack);
				ArrayList<Integer> found = new ArrayList<Integer>();
				for (TreeNode node : path) {
					found.add(node.val);
				}
				result.add(found);
			}
			TreeNode child = null;
			root = stack.peek();
			while (!stack.isEmpty() && root != null
					&& (root.right == null || root.right == child)) {
				child = stack.pop();
				accu -= child.val;
				if (!stack.isEmpty()) {
					root = stack.peek();
				} else {
					root = null;
				}
			}
			root = root == null ? null : root.right;
		}
		return result;
	}

	private void pathSumHelper(TreeNode root, int sum, ArrayList<Integer> path,
			ArrayList<ArrayList<Integer>> result) {
		if (root == null) {
			return;
		}
		path.add(root.val);
		if (root.left == null && root.right == null && root.val == sum) {
			ArrayList<Integer> found = new ArrayList<Integer>(path);
			result.add(found);
		} else {
			pathSumHelper(root.left, sum - root.val, path, result);
			pathSumHelper(root.right, sum - root.val, path, result);
		}
		path.remove(path.size() - 1);
	}

	// Binary Tree Preorder Traversal
	public ArrayList<Integer> preorderTraversal(TreeNode root) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		Stack<TreeNode> stack = new Stack<TreeNode>();
		while (root != null) {
			while (root != null) {
				stack.push(root);
				result.add(root.val);
				root = root.left != null ? root.left : root.right;
			}
			TreeNode child = null;
			root = stack.peek();
			while (!stack.isEmpty() && root != null
					&& (root.right == null || root.right == child)) {
				child = stack.pop();
				if (!stack.isEmpty()) {
					root = stack.peek();
				} else {
					root = null;
				}
			}
			root = root == null ? null : root.right;
		}
		return result;
	}

	// Binary Tree Postorder Traversal
	public ArrayList<Integer> postorderTraversal(TreeNode root) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		Stack<TreeNode> stack = new Stack<TreeNode>();
		while (root != null) {
			while (root != null) {
				stack.push(root);
				root = root.left != null ? root.left : root.right;
			}
			TreeNode child = null;
			root = stack.peek();
			while (!stack.isEmpty() && root != null
					&& (root.right == null || root.right == child)) {
				child = stack.pop();
				result.add(child.val);
				if (!stack.isEmpty()) {
					root = stack.peek();
				} else {
					root = null;
				}
			}
			root = root == null ? null : root.right;
		}
		return result;
	}

	// Subsets II
	public ArrayList<ArrayList<Integer>> subsetsWithDup(int[] num) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<ArrayList<Integer>>();
		if (num == null) {
			return result;
		}
		ArrayList<Integer> empty = new ArrayList<Integer>();
		result.add(empty);
		if (num.length == 0) {
			return result;
		}
		Arrays.sort(num);
		int partialSize = 0;
		for (int i = 0; i < num.length; i++) {
			int start = (i == 0 || num[i] != num[i - 1]) ? 0 : result.size()
					- partialSize;
			int end = result.size() - 1;
			for (int j = start; j <= end; j++) {
				ArrayList<Integer> al = new ArrayList<Integer>(result.get(j));
				al.add(num[i]);
				result.add(al);
			}
			partialSize = end - start + 1;
		}
		return result;
	}

	public static void main(String[] args) {
		// Solution s = new Solution();
	}
}