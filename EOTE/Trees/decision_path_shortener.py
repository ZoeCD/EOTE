from EOTE.Protocols.Trees import PathShortener
from typing import List
import re


class PathShortenerMixedData(PathShortener):
    def shorten_path(self, all_features: List[str], current_features: List[str], path: str) -> str:
        new_path = "If"
        not_first = False

        for i in range(len(current_features)):
            feature = current_features[i]
            feature_type = 'Numerical' if feature in all_features else 'Nominal'
            if feature_type == 'Nominal':
                feature = self.__get_real_feature_name(feature)
                preposition = re.findall(f"([(]{feature}\w*.....\w*.\w*[)])", path)
            else:
                preposition = re.findall(f"([(]{feature}....\w*.\w*[)])", path)

            if len(preposition) > 0 and feature not in new_path:
                if feature_type == 'Numerical':
                    new_string = self.__shorten_numerical_feature(feature, preposition)
                else:
                    new_string = self.__shorten_nominal_feature(feature, preposition)
                if not_first:
                    new_path += "AND" + new_string
                else:
                    not_first = True
                    new_path += new_string

        result = re.search("then.*", path)
        if not result:
            return path #
        new_path += result.group(0)
        return new_path

    def __shorten_numerical_feature(self, feature: str, preposition: str) -> str:
        values_less_than = re.findall(f"(?<=≤ ).?[0-9]+\.?[0-9]*", str(preposition))
        values_greater_than = re.findall(f"(?<=> ).?[0-9]+\.?[0-9]*", str(preposition))
        min_num = None
        max_num = None
        if len(values_less_than) >= 1:
            max_num = min(values_less_than)
        if len(values_greater_than) >= 1:
            min_num = max(values_greater_than)

        if max_num and min_num:
            new_string = f" ({min_num} < {feature} ≤ {max_num}) "
        elif min_num:
            new_string = f" ({feature} > {min_num}) "
        else:
            new_string = f" ({feature} ≤ {max_num}) " 
        
        return new_string

    def __shorten_nominal_feature(self, feature: str, preposition: str) -> str:
        values_not = re.findall(f"(?<=_)\w*(?= ≤)", str(preposition))
        values_is = re.findall(f"(?<=_)\w*(?= >)", str(preposition))
        new_string = ""

        if len(values_is) >= 1:
            return f" ({feature} is " + str(values_is[0]) + ") "
        elif len(values_not) <= 1:
            return f" ({feature} ≠ " + str(values_not[0]) + ") "
        else:
            values = ", ".join(values_not)
            new_string += f" ({feature} ≠ " + values + ") "
            return new_string

    def __get_real_feature_name(self, feature: str) -> str:
        feature = feature.split('_')
        name = '_'.join([str(elem) for elem in feature[:-1]])
        name = f"{name}"
        return name