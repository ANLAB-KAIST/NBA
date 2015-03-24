FromInput()->
CheckIPHeader() -> branch :: RandomWeightedBranch(60, 40);
pathA :: L2Forward(method echoback) -> ToOutput();
pathB :: Discard();
branch[0] -> pathA;
branch[1] -> pathB;
