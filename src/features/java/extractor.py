from collections import defaultdict
from typing import *

from common.var_example import VarExample
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


class JavaVarExamplesExtractor:
    @staticmethod
    def from_source_file(source_file: str) -> List[VarExample]:
        ast = JavaAst(source_file)

        # Get methods
        methods_nodes = list(
            ast.get_nodes(types=NODE_ELEMENT_TYPES, content="METHOD")
        )

        # Map local variables and parameters to a numeric id
        var_map = defaultdict(int)
        var_num = 0
        var_nodes = ast.get_nodes(types=NODE_VAR_TYPES)
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
                    idt_node.pos[0] >= method_node.pos[0]
                    and idt_node.pos[1] <= method_node.pos[1]
                    for method_node in methods_nodes
                ) or any(
                    idt_node.pos[0] >= memb_node.pos[0]
                    and idt_node.pos[1] <= memb_node.pos[1]
                    for memb_node in memb_nodes
                ):
                    # Ignore identifiers that appear outside the method or as
                    # class fields.
                    skip = True
                    break
            if skip:
                continue

            for idt_node in idt_nodes:
                var_map[idt_node.pos[0]] = var_num + 1
            var_num += 1
        del memb_nodes

        # Create one example for each method
        examples = []
        for method_node in methods_nodes:
            token_nodes = list(
                ast.get_nodes(types=NODE_TOKEN_TYPES, pos=method_node.pos)
            )
            tokens = list(map(lambda n: n.content, token_nodes))
            mask = list(map(lambda n: var_map[n.pos[0]], token_nodes))
            examples.append(VarExample(tokens, mask))

        return examples
