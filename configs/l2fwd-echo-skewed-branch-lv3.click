FromInput -> lv1 :: RandomWeightedBranch({0},{1});
out :: L2Forward(method echoback) -> ToOutput();
lv1[0] -> lv2 :: RandomWeightedBranch({0},{1});
lv1[1] -> out;
lv2[0] -> lv3 :: RandomWeightedBranch({0},{1});
lv2[1] -> out;
lv3[0] -> out;
lv3[1] -> out;
