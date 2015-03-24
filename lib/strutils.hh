#ifndef __NSHADER_STRUTILS_HH__
#define __NSHADER_STRUTILS_HH__

#include <cstring>
#include <string>

namespace nshader {

/**
 * This function assumes that parent_path, current_path have enough buffer space.
 * TODO: make it safer
 */
inline int split_filename(const char *fullpath, char *parent_path, char *current_path)
{
    int found = -1;
    int full_len = strlen(fullpath);
    int i;
    for (i = full_len - 1; i >= 0; i--) {
        if (fullpath[i] == '/') {
            found = i;
            break;
        }
    }
    if (found == -1)
        return -1;
    if (parent_path != NULL) {
        strncpy(parent_path, &fullpath[0], found);
        parent_path[found] = '\0';
    }
    if (current_path != NULL) {
        strncpy(current_path, &fullpath[found + 1], full_len - found - 1);
        current_path[full_len - found - 1] = '\0';
    }
    return 0;
}

inline std::string string_upper(const std::string &orig)
{
    std::string temp;
    for(unsigned k=0; k<orig.size(); k++)
    {
        char single = orig[k];
        if(single > 'Z')
            single = single - 'a' + 'A';
        temp += single;
    }
    return temp;
}

}

#endif

// vim: ts=8 sts=4 sw=4 et
