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

	public static void main(String[] args) {
	}
}