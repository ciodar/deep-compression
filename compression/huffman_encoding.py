import logging

import numpy as np

log = logging.getLogger(__name__)


# Node of a Huffman Tree
class Node:
    def __init__(self, probability, index, left=None, right=None):
        # probability of the symbol
        self.probability = probability
        # the symbol
        self.index = index
        # the left node
        self.left = left
        # the right node
        self.right = right
        # the tree direction (0 or 1)
        self.code = ''


class HuffmanEncode:
    def __init__(self, bits=5):
        self.symbols, self.codes = {}, {}
        self.initial_bits = bits

    """ Calculates frequency of every index in data"""

    def frequency(self, data):
        indices, frequencies = np.unique(data, return_counts=True)
        return indices, frequencies

    """ Encodes the symbols by visiting the Huffman Tree """

    def codify(self, node, value=''):
        # a huffman code for current node
        newValue = value + str(node.code)

        if node.left:
            self.codify(node.left, newValue)
        if node.right:
            self.codify(node.right, newValue)

        if not node.left and not node.right:
            self.codes[node.index] = newValue
        return self.codes

    def get_encoded(self, data, coding):
        out = [coding[e] for e in data]
        return ''.join([str(item) for item in out])

    """ A supporting function in order to calculate the space difference between compressed and non compressed data"""

    def get_gain(self, data, coding):
        # total bit space to store the data before compression
        n_data = len(data)
        before = n_data * self.initial_bits
        after = 0
        symbols = coding.keys()
        for symbol in symbols:
            count = np.count_nonzero(data == symbol)
            # calculating how many bit is required for that symbol in total
            after += count * len(coding[symbol])
            #log.debug(f"  Symbol: {symbol} | count: {count:.0f} | coding length: {len(coding[symbol])}")
        log.debug("  Space usage before huffman encoding for {:.0f} values (in bits): {:.0f}".format(n_data, before))
        log.debug("  Space usage after huffman encoding for {:.0f} values (in bits): {:.0f}".format(n_data, after))
        log.info("  Average bits: {:.1f}".format(after / n_data))
        return after, after / n_data

    @classmethod
    def encode(cls, data, bits=5):
        huffman = cls(bits=bits)
        symbols, frequencies = huffman.frequency(data)
        # print("symbols: ", symbols)
        # print("frequencies: ", the_probabilities)

        nodes = []

        # converting symbols and probabilities into huffman tree nodes
        for s, f in zip(symbols, frequencies):
            nodes.append(Node(f, s))

        while len(nodes) > 1:
            # sorting all the nodes in ascending order based on their probability
            nodes = sorted(nodes, key=lambda x: x.probability)
            # for node in nodes:
            #      print(node.index, node.prob)

            # picking two smallest nodes
            right = nodes[0]
            left = nodes[1]

            left.code = 0
            right.code = 1

            # combining the 2 smallest nodes to create new node
            new = Node(left.probability + right.probability, left.index + right.index, left, right)

            nodes.remove(left)
            nodes.remove(right)
            nodes.append(new)

        huffmanEncoding = huffman.codify(nodes[0])
        # print("symbols with codes", huffmanEncoding)
        tot_size, avg_bits = huffman.get_gain(data, huffmanEncoding)
        # encoded = huffman.get_encoded(data, huffmanEncoding)
        return tot_size, avg_bits

    def decode(self, encoded, tree):
        treeHead = tree
        decoded = []
        for x in encoded:
            if x == '1':
                huffmanTree = huffmanTree.right
            elif x == '0':
                huffmanTree = huffmanTree.left
            try:
                if huffmanTree.left.index == None and huffmanTree.right.index == None:
                    pass
            except AttributeError:
                decoded.append(huffmanTree.index)
                huffmanTree = treeHead

        string = ''.join([str(item) for item in decoded])
        return string
