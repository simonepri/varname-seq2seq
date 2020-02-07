import re
from collections import defaultdict
from typing import *

from utils.strings import multiple_replace
from features.java.ast import (
    JavaAst,
    JavaAstNode,
    JavaAstEdgeType,
    JavaAstNodeType,
)


NODE_ELEMENT_TYPES = [JavaAstNodeType.AST_ELEMENT]
NODE_VAR_TYPES = [JavaAstNodeType.SYMBOL_VAR]
NODE_TOKEN_TYPES = [JavaAstNodeType.TOKEN, JavaAstNodeType.IDENTIFIER_TOKEN]
NODE_FAKE_TYPES = [JavaAstNodeType.FAKE_AST]
EDGE_CHILD_TYPES = [JavaAstEdgeType.AST_CHILD]
EDGE_ASYM_TYPES = [JavaAstEdgeType.ASSOCIATED_SYMBOL]


class JavaLocalVarExamples:
    def __init__(self, examples: List[Tuple[List[str], List[int]]]):
        self.examples = examples

    def save(self, file_path: str):
        with open(file_path, "w+") as f:
            for tokens, masks in self.examples:
                example_builder = []
                for token, varid in zip(tokens, masks):
                    token = self.__encode_token(token)
                    example_builder.append("%s:%d" % (token, varid))
                example_str = "\t".join(example_builder)
                print(example_str, file=f)

    @staticmethod
    def load(file_path: str):
        examples = []
        with open(file_path, "r+") as f:
            tokens, masks = [], []
            for line in f:
                parts = line.split("\t")
                for part in parts:
                    token, _, varid = part.rpartition(":")
                    token = self.__decode_token(token)
                    tokens.append(token)
                    masks.append(varid)
            examples.append((tokens, masks))
        return JavaLocalVarExamples(examples)

    @staticmethod
    def from_source_file(source_file: str) -> "JavaLocalVarExamples":
        ast = JavaAst(source_file)

        # Map local variables to a numeric id
        var_map = defaultdict(int)
        var_num = 0
        var_nodes = ast.get_nodes(types=NODE_VAR_TYPES)
        body_nodes = list(ast.get_nodes(types=NODE_FAKE_TYPES, content="BODY"))
        memb_nodes = list(
            ast.get_nodes(types=NODE_ELEMENT_TYPES, content="MEMBER_SELECT")
        )
        for var_node in var_nodes:
            edges = ast.get_edges(var_node, types=EDGE_ASYM_TYPES)
            idt_nodes = list(map(lambda e: e.nodes[1], edges))

            # Filter local variables.
            skip = False
            for idt_node in idt_nodes:
                # TODO: do each query in log(n)
                if not any(
                    idt_node.pos[0] >= body_node.pos[0]
                    and idt_node.pos[1] <= body_node.pos[1]
                    for body_node in body_nodes
                ) or any(
                    idt_node.pos[0] >= memb_node.pos[0]
                    and idt_node.pos[1] <= memb_node.pos[1]
                    for memb_node in memb_nodes
                ):
                    # Ignore variables that appear outside the body or as class
                    # fields.
                    skip = True
                    break
            if skip:
                continue

            for idt_node in idt_nodes:
                var_map[idt_node.pos[0]] = var_num + 1
            var_num += 1
        del body_nodes, memb_nodes

        # Create one example for each method
        examples = []
        methods_nodes = ast.get_nodes(
            types=NODE_ELEMENT_TYPES, content="METHOD"
        )
        for method_node in methods_nodes:
            token_nodes = list(
                ast.get_nodes(types=NODE_TOKEN_TYPES, pos=method_node.pos)
            )
            tokens = list(map(lambda n: n.content, token_nodes))
            mask = list(map(lambda n: var_map[n.pos[0]], token_nodes))
            examples.append((tokens, mask))

        return JavaLocalVarExamples(examples)

    @classmethod
    def __encode_token(cls, token: str):
        map = {"\n": r"\n", "\r": r"\r", "\t": r"\t"}
        return multiple_replace(map, token)

    @classmethod
    def __decode_token(cls, token: str):
        map = {r"\n": "\n", r"\r": "\r", r"\t": "\t"}
        return multiple_replace(map, token)
