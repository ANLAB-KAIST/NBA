echo 'Fusing and updating gtest sources/headers for NBA...'
echo 'NOTE: You need to run "git pull" in 3rdparty/googletest manually first.'
TMPDIR=/tmp/nba-gtest-update
NBADIR=..
mkdir -p $TMPDIR
python2 googletest/googletest/scripts/fuse_gtest_files.py $TMPDIR
cp googletest/googletest/src/gtest_main.cc $TMPDIR/gtest
# Replace relative include path to absolute path
sed -i 's/^#include "gtest\/gtest.h"/#include <gtest\/gtest.h>/' $TMPDIR/gtest/*.cc
cp $TMPDIR/gtest/gtest-all.cc $NBADIR/src/lib/gtest
cp $TMPDIR/gtest/gtest_main.cc $NBADIR/src/lib/gtest
cp $TMPDIR/gtest/gtest.h $NBADIR/include/gtest
rm -rf $TMPDIR
