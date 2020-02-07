import os
import heapq
import itertools
from collections import defaultdict
from subprocess import DEVNULL, STDOUT, check_call, CalledProcessError
from typing import *

import utils.bisect as bisect
from proto.graph_pb2 import Graph
from proto.graph_pb2 import FeatureNode as JavaAstNodeType
from proto.graph_pb2 import FeatureEdge as JavaAstEdgeType
from utils.files import walk_files
from utils.download import download_url


class JavaAstNode:
    def __init__(
        self, idx: int, type: int, content: str, startPos: int, endPos: int
    ):
        self.idx = idx
        self.type = type
        self.content = content
        self.pos = (startPos, endPos)

    def __repr__(self):
        return str(
            {
                "idx": self.idx,
                "type": self.type,
                "content": self.content,
                "pos": self.pos,
            }
        )


class JavaAstEdge:
    def __init__(
        self, source: JavaAstNode, type: int, destination: JavaAstNode
    ):
        self.type = type
        self.nodes = (source, destination)

    def __repr__(self):
        return str((self.nodes[0].idx, self.type, self.nodes[1].idx))


class JavaAst:
    SETUP = False
    JAVAC_BIN_PATH = os.getenv("JAVAC_BIN_PATH", "javac")
    JAVAC_EXTRACTOR_DOWNLOAD_URL = (
        "https://storage.googleapis.com"
        + "/features-javac/features-javac-extractor-latest.jar"
    )
    AST_CACHE_DIR = os.getenv("AST_CACHE_DIR", ".cache/java_ast")
    AST_EXTRACTOR_DIR = os.path.join(AST_CACHE_DIR, "bin")
    AST_EXTRACTOR_PATH = os.path.join(
        AST_EXTRACTOR_DIR, "features-javac-extractor.jar"
    )
    AST_PROTO_DIR = os.path.join(AST_CACHE_DIR, "proto")

    def __init__(self, source_file_path: str):
        if not JavaAst.SETUP:
            raise Exception(
                "The AST Parser is not initalized. "
                + "Run JavaAst.setup() to download the parser."
            )
        proto = JavaAst.__get_proto(source_file_path)
        graph = Graph()
        graph.ParseFromString(proto)

        id_to_idx = {}

        self.nodes = []
        self.adj = [[] for _ in graph.node]
        self.nodes_by_type = defaultdict(list)

        nodes = list(graph.node)
        nodes.reverse()
        nodes.sort(
            key=lambda n: (n.startPosition, -n.endPosition), reverse=True
        )
        edges = list(graph.edge)
        del graph

        while len(nodes) > 0:
            idx = len(self.nodes)
            node = nodes.pop()
            ast_node = JavaAstNode(
                idx,
                node.type,
                node.contents,
                node.startPosition,
                node.endPosition,
            )
            id_to_idx[node.id] = idx
            self.nodes.append(ast_node)

        for node in self.nodes:
            self.nodes_by_type[node.type].append(node)

        while len(edges) > 0:
            edge = edges.pop()
            srcIdx = id_to_idx[edge.sourceId]
            dstIdx = id_to_idx[edge.destinationId]
            ast_edge = JavaAstEdge(
                self.nodes[srcIdx], edge.type, self.nodes[dstIdx]
            )
            self.adj[srcIdx].append(ast_edge)

    def get_nodes(
        self,
        types: Optional[List[int]] = None,
        content: Optional[str] = None,
        pos: Tuple[Optional[int], Optional[int]] = (None, None),
    ) -> Iterable[JavaAstNode]:
        parts = []
        if types is None:
            splits = [self.nodes]
        else:
            splits = [self.nodes_by_type[type] for type in types]
        for split in splits:
            start, stop = None, None
            if pos[0] is not None:
                start = bisect.index_ge(split, pos[0], key=lambda n: n.pos[0])
            if pos[1] is not None:
                stop = (
                    bisect.index_le(split, pos[1], key=lambda n: n.pos[1]) + 1
                )
            part = itertools.islice(split, start, stop)
            if content is not None:
                part = filter(lambda n: n.content == content, part)
            parts.append(part)
        return heapq.merge(*parts, key=lambda n: (n.pos[0], -n.pos[1]))

    def get_edges(
        self,
        node: JavaAstNode,
        types: Optional[List[int]] = None,
        dest_content: Optional[str] = None,
        dest_types: Optional[List[int]] = None,
    ) -> Iterable[JavaAstEdge]:
        edges = self.adj[node.idx]
        if types is not None:
            edges = filter(lambda e: e.type in types, edges)
        if dest_content is not None:
            edges = filter(lambda e: e.nodes[1].content == dest_content, edges)
        if dest_types is not None:
            edges = filter(lambda e: e.nodes[1].type in dest_types, edges)
        return edges

    @classmethod
    def setup(cls, progress: bool = False) -> None:
        if not os.path.isfile(cls.AST_EXTRACTOR_PATH):
            os.makedirs(cls.AST_EXTRACTOR_DIR, exist_ok=True)
            download_url(
                cls.JAVAC_EXTRACTOR_DOWNLOAD_URL,
                cls.AST_EXTRACTOR_PATH,
                progress=progress,
            )
        os.makedirs(cls.AST_PROTO_DIR, exist_ok=True)
        cls.SETUP = True

    @classmethod
    def cache_files(cls, file_paths: List[str]):
        """Use the javac compiler and features extration plugin to extract the
        ASTs.
        """
        already_cached = True
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                raise IOError("File not found: %s" % file_path)
            if not cls.file_cached(file_path):
                already_cached = False
        if already_cached:
            return

        cls.__run_extractor(file_paths)

        missing_files = []
        for file_path in file_paths:
            gen_proto_file_path = file_path + ".proto"
            if not os.path.isfile(gen_proto_file_path):
                missing_files.append(file_path)
                continue
            proto_file_path = cls.__cache_path_for_file(file_path)
            os.rename(gen_proto_file_path, proto_file_path)
        if len(missing_files) > 0:
            raise IOError("Files not cached: %s" % missing_files)

    @classmethod
    def file_cached(cls, file_path: str) -> bool:
        proto_file_path = cls.__cache_path_for_file(file_path)
        if not os.path.isfile(proto_file_path):
            return False
        if os.path.getmtime(proto_file_path) < os.path.getmtime(file_path):
            return False
        return True

    @classmethod
    def __cache_path_for_file(cls, file_path: str) -> str:
        slug = os.path.relpath(file_path.replace("/", ":"))
        return os.path.join(cls.AST_PROTO_DIR, slug + ".proto")

    @classmethod
    def __get_proto(cls, file_path: str) -> str:
        cls.cache_files([file_path])
        proto_file_path = cls.__cache_path_for_file(file_path)
        with open(proto_file_path, "rb") as f:
            return f.read()

    @classmethod
    def __run_extractor(cls, file_paths: List[str]) -> None:
        javac_path = cls.JAVAC_BIN_PATH
        jar_path = cls.AST_EXTRACTOR_PATH
        cmd = [
            javac_path,
            "-cp",
            jar_path,
            "-Xplugin:FeaturePlugin",
            " ".join(file_paths),
        ]
        try:
            check_call(cmd, stdout=DEVNULL, stderr=STDOUT)
        except CalledProcessError as e:
            # This might happen if the files contain unresolved dependencies.
            pass
