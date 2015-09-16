//FromInput -> lv1_head :: RandomWeightedBranch({0}, {1});
//lv1_head[0] -> L2Forward(method {2}) -> ToOutput();
//lv1_head[1] -> L2Forward(method {2}) -> ToOutput();
FromInput -> lv1_head :: RandomWeightedBranch(0.3, 0.7);
lv1_head[0] -> L2Forward(method echoback) -> ToOutput();
lv1_head[1] -> L2Forward(method echoback) -> ToOutput();
