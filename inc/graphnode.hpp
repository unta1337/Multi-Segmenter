#ifndef __GRAPHNODE_H
#define __GRAPHNODE_H

namespace FaceGraph {
class GraphNode {
  public:
    int triangles;
    GraphNode* link;

    ~GraphNode() {
        delete (link);
    }
};
} // namespace FaceGraph

#endif
