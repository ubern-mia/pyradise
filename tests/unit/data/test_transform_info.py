from pyradise.data.taping import TransformInfo, TransformTape


def test_get_subclasses():
    tra_info = TransformInfo("name_1", None, None, None, None, None, None)

    tra_info._get_subclasses(TransformInfo)


# def get_filter():
#     from pyradise.process import Filter
#
#     subclasses = self._get_subclasses(Filter)
#     return subclasses.get(self.name)(**self.filter_args)
#
#
# def get_params():
#     return self.params
#
#
# def get_image_properties():
#     if pre_transform:
#         return self.pre_transform_image_properties
#     return self.post_transform_image_properties
#
#
# def add_data():
#     self.additional_data[key] = value
#
#
# def get_data():
#     return self.additional_data.get(key, None)
#
#
# def get_transform():
#     if self.transform is not None:
#         if inverse:
#             return self.transform.GetInverse()
#         return self.transform
#
#     # check if the image origin and direction have changed
#     num_dims = len(self.pre_transform_image_properties.size)
#     if self.pre_transform_image_properties.has_equal_origin_direction(self.post_transform_image_properties):
#         transform = sitk.AffineTransform(num_dims)
#         transform.SetIdentity()
#         return transform
#
#     else:
#         transform = sitk.AffineTransform(num_dims)
#         transform.SetIdentity()
#
#         # compute the translation
#         post_origin = self.post_transform_image_properties.origin
#         pre_origin = self.pre_transform_image_properties.origin
#         translation = list(np.array(post_origin) - np.array(pre_origin))
#
#         # compute the rotation
#         post_direction = np.array(self.post_transform_image_properties.direction).reshape(num_dims, num_dims)
#         pre_direction = np.array(self.pre_transform_image_properties.direction).reshape(num_dims, num_dims)
#         rotation = np.matmul(np.linalg.inv(pre_direction), post_direction)
#         rotation = list(rotation.reshape(-1))
#
#         # set the transform parameters
#         transform.SetParameters(rotation + translation)
#
#         # return the inverted or the original transform
#         if inverse:
#             transform = transform.GetInverse()
#         return transform
