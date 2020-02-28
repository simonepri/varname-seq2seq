from collections import defaultdict
from typing import *  # pylint: disable=W0401,W0614

from features.examples import VarExample
from features.java.ast import (
    JavaAst,
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
            ast.get_nodes(node_types=NODE_ELEMENT_TYPES, content="METHOD")
        )

        # Map local variables and parameters to a numeric id
        var_map = defaultdict(int)
        var_num = 0
        var_nodes = ast.get_nodes(node_types=NODE_VAR_TYPES)
        param_nodes = list(
            ast.get_nodes(node_types=NODE_FAKE_TYPES, content="PARAMETERS")
        )
        memb_nodes = list(
            ast.get_nodes(
                node_types=NODE_ELEMENT_TYPES, content="MEMBER_SELECT"
            )
        )
        # For each node marked as variable
        for var_node in var_nodes:
            edges = ast.get_edges(var_node, edge_types=EDGE_ASYM_TYPES)
            # Get all the identifiers
            idt_nodes = list(map(lambda e: e.nodes[1], edges))

            include = False

            # Include identifiers that appear as method arguments.
            if not include:
                for idt_node in idt_nodes:
                    # TODO: do the query in log(n)
                    appears_as_parameter = any(
                        idt_node.pos[0] >= params_node.pos[0]
                        and idt_node.pos[1] <= params_node.pos[1]
                        for params_node in param_nodes
                    )
                    if appears_as_parameter:
                        # Always include method parameters
                        include = True
                        break

            # Exclude identifiers that are fields or this.
            if not include:
                include = True
                for idt_node in idt_nodes:
                    # TODO: do each query in log(n)
                    appears_in_a_method = any(
                        idt_node.pos[0] >= method_node.pos[0]
                        and idt_node.pos[1] <= method_node.pos[1]
                        for method_node in methods_nodes
                    )
                    part_of_a_member_select = any(
                        idt_node.pos[0] >= memb_node.pos[0]
                        and idt_node.pos[1] <= memb_node.pos[1]
                        for memb_node in memb_nodes
                    )
                    if not appears_in_a_method or part_of_a_member_select:
                        # Ignore identifiers that appear outside the method or
                        # as class fields.
                        include = False
                        break

            if not include:
                continue

            for idt_node in idt_nodes:
                var_map[idt_node.pos[0]] = var_num + 1
            var_num += 1
        del memb_nodes

        # Create one example for each method
        examples = []
        for method_node in methods_nodes:
            token_nodes = list(
                ast.get_nodes(node_types=NODE_TOKEN_TYPES, pos=method_node.pos)
            )
            tokens = list(map(lambda n: n.content, token_nodes))
            mask = list(map(lambda n: var_map[n.pos[0]], token_nodes))
            examples.append(VarExample(tokens, mask))

        return examples
