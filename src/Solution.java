import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
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
	public void connect(TreeLinkNode root) {
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
	public int maxProfit(int[] prices) {
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
	public double findMedianSortedArrays(int A[], int B[]) {
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
		if (compare <= 0 || A[mid] > B[compare - 1]) {
			return findMedianSortedArraysHelper(A, B, left, mid - 1);
		} else {
			return findMedianSortedArraysHelper(A, B, mid + 1, right);
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
	public ArrayList<String> wordBreak(String s, Set<String> dict) {
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

	public static void main(String[] args) {
		// Solution s = new Solution();
	}
}