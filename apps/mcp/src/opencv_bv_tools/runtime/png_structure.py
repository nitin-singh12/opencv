# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

"""Bounded PNG structural checks, not a full PNG/ICC/EXIF conformance validator."""
from dataclasses import dataclass
import struct
import zlib

SIGNATURE = b'\x89PNG\r\n\x1a\n'
KNOWN_ANCILLARY = {b'tRNS',b'cHRM',b'gAMA',b'iCCP',b'sRGB',b'sBIT',b'bKGD',b'hIST',
                   b'pHYs',b'tEXt',b'zTXt',b'iTXt',b'tIME',b'eXIf',b'cICP',b'mDCV',b'cLLI',b'sPLT'}


class PNGInputError(ValueError):
    pass


class PNGResourceError(PNGInputError):
    pass


@dataclass(frozen=True)
class PNGStructure:
    chunks: tuple
    width: int
    height: int
    depth: int
    color: int
    interlace: int

    @property
    def representation(self):
        return tuple((kind,value) for kind,value,_ in self.chunks if kind in {b'IHDR',b'PLTE',b'tRNS'})

    @property
    def non_idat(self):
        return tuple(raw for kind,_,raw in self.chunks if kind != b'IDAT')


def scanline_bytes(width, height, depth, color, interlace):
    channels = {0:1,2:3,3:1,4:2,6:4}[color]
    passes = ((0,0,1,1),) if not interlace else (
        (0,0,8,8),(4,0,8,8),(0,4,4,8),(2,0,4,4),(0,2,2,4),(1,0,2,2),(0,1,1,2))
    size = 0
    for x,y,dx,dy in passes:
        w=max(0,(width-x+dx-1)//dx);h=max(0,(height-y+dy-1)//dy)
        if w and h:size += h*(1+(w*channels*depth+7)//8)
    return size


def _inflate(data, limit, *, resource=False):
    decoder=zlib.decompressobj()
    try:
        output=decoder.decompress(data,limit+1)
    except zlib.error as exc:
        raise PNGInputError('invalid_zlib_stream') from exc
    if len(output)>limit:
        raise (PNGResourceError if resource else PNGInputError)('decompressed_size_limit')
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise PNGInputError('invalid_or_extra_zlib_stream')
    return output


def _metadata_expansion(kind, value, remaining):
    if kind not in {b'iCCP',b'zTXt',b'iTXt'}:return 0
    keyword,separator,rest=value.partition(b'\0')
    if not separator or not 1<=len(keyword)<=79:raise PNGInputError('invalid_metadata')
    if kind in {b'iCCP',b'zTXt'}:
        if not rest or rest[0]!=0:raise PNGInputError('invalid_metadata_compression')
        return len(_inflate(rest[1:],remaining,resource=True))
    if len(rest)<2 or rest[0] not in (0,1) or rest[1]!=0:raise PNGInputError('invalid_metadata_compression')
    flag=rest[0];parts=rest[2:].split(b'\0',2)
    if len(parts)!=3:raise PNGInputError('invalid_itxt')
    return len(_inflate(parts[2],remaining,resource=True)) if flag else 0


def parse_png(data, limits):
    if not data.startswith(SIGNATURE):raise PNGInputError('not_png')
    if len(data)>limits['max_input_bytes']:raise PNGResourceError('input_byte_limit')
    position=8;chunks=[];seen=set();idat_started=False;idat_ended=False
    width=height=depth=color=interlace=None;palette_entries=0;metadata=0
    while position<len(data):
        if len(chunks)>=limits['max_chunks']:raise PNGResourceError('chunk_limit')
        if position+12>len(data):raise PNGInputError('truncated_chunk')
        length=int.from_bytes(data[position:position+4],'big')
        if length>0x7fffffff or position+12+length>len(data):raise PNGInputError('invalid_chunk_length')
        kind=data[position+4:position+8];value=data[position+8:position+8+length]
        raw=data[position:position+12+length];position+=length+12
        if any(not (65<=c<=90 or 97<=c<=122) for c in kind) or kind[2]&32:
            raise PNGInputError('invalid_chunk_type')
        expected=int.from_bytes(raw[-4:],'big')
        if zlib.crc32(kind+value)&0xffffffff != expected:raise PNGInputError('crc_mismatch')
        if not chunks and kind!=b'IHDR':raise PNGInputError('ihdr_must_be_first')
        if kind in {b'IHDR',b'PLTE',b'IEND',b'tRNS'} and kind in seen:
            raise PNGInputError('duplicate_structural_chunk')
        if kind==b'IHDR':
            if length!=13:raise PNGInputError('invalid_ihdr')
            width,height,depth,color,compression,filter_method,interlace=struct.unpack('>IIBBBBB',value)
            if not width or not height or width>0x7fffffff or height>0x7fffffff:raise PNGInputError('invalid_dimensions')
            if width*height>limits['max_pixels']:raise PNGResourceError('input_pixel_limit')
            if max(width,height)>limits['max_dimension']:raise PNGResourceError('input_dimension_limit')
            if depth>8:raise PNGInputError('high_bit_depth_not_supported')
            if color not in (0,2,3,4,6) or depth not in ({1,2,4,8} if color in (0,3) else {8}):
                raise PNGInputError('unsupported_color_depth')
            if compression!=0 or filter_method!=0 or interlace not in (0,1):raise PNGInputError('unsupported_ihdr_method')
        elif kind==b'PLTE':
            if idat_started or color in (0,4) or not length or length%3 or length>768:
                raise PNGInputError('invalid_palette')
            palette_entries=length//3
            if color==3 and palette_entries>2**depth:raise PNGInputError('invalid_palette')
        elif kind==b'tRNS':
            if idat_started or color in (4,6):raise PNGInputError('invalid_transparency')
            if color==3 and not 0<length<=palette_entries:raise PNGInputError('invalid_transparency')
            if color==0 and length!=2 or color==2 and length!=6:raise PNGInputError('invalid_transparency')
        elif kind==b'IDAT':
            if idat_ended or (color==3 and b'PLTE' not in seen):raise PNGInputError('invalid_idat_order')
            idat_started=True
        elif kind==b'IEND':
            if length or not idat_started or position!=len(data):raise PNGInputError('invalid_iend_or_trailing_data')
        elif kind in {b'acTL',b'fcTL',b'fdAT'}:
            raise PNGInputError('animation_not_supported')
        elif not kind[0]&32:
            raise PNGInputError('unknown_critical_chunk')
        elif kind in {b'caBX',b'iDOT',b'dSIG'} or (kind not in KNOWN_ANCILLARY and not kind[3]&32):
            raise PNGInputError('unsupported_unsafe_metadata')
        if idat_started and kind!=b'IDAT':idat_ended=True
        if kind not in {b'IHDR',b'PLTE',b'IDAT',b'IEND',b'tRNS'}:
            metadata+=length
            if metadata>limits['max_metadata_bytes']:raise PNGResourceError('metadata_limit')
            metadata+=_metadata_expansion(kind,value,limits['max_metadata_bytes']-metadata)
            if metadata>limits['max_metadata_bytes']:raise PNGResourceError('metadata_limit')
        seen.add(kind);chunks.append((kind,value,raw))
    if not chunks or chunks[-1][0]!=b'IEND':raise PNGInputError('missing_iend')
    compressed=b''.join(v for k,v,_ in chunks if k==b'IDAT')
    expected=scanline_bytes(width,height,depth,color,interlace)
    if len(_inflate(compressed,expected))!=expected:raise PNGInputError('wrong_scanline_size')
    return PNGStructure(tuple(chunks),width,height,depth,color,interlace)


def transplant_idat(source, candidate):
    if source.representation!=candidate.representation:
        raise PNGInputError('optimizer_changed_representation')
    blocks=[raw for kind,_,raw in candidate.chunks if kind==b'IDAT']
    output=[SIGNATURE];written=False
    for kind,_,raw in source.chunks:
        if kind==b'IDAT':
            if not written:output.extend(blocks);written=True
        else:output.append(raw)
    return b''.join(output)
