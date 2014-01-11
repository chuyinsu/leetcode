import java.util.ArrayList;
import java.util.Arrays;
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

	public static void main(String[] args) {
	}
}